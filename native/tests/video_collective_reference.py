"""Test-only source collection and logical-rank replay on one device.

The empty-peer adapter isolates collection promotion. The partitioned adapter
also dispatches real original neural rank-local sessions. Neither adapter is a
physical distributed execution test. Collection bytecode uses copied globals,
so torch.distributed is never patched globally.
"""
import inspect
import types

import numpy as np
import torch


def install_empty_peer_replay(model, ranks=2):
    if ranks < 2:
        raise ValueError("replay requires at least two ranks")
    original = model.run_tracker_propagation
    method = original.__func__
    signature = inspect.signature(original)
    records = []

    def replay(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        masks, scores = original(*args, **kwargs)
        ids = inputs["tracker_metadata_prev"]["obj_ids_per_gpu"][model.rank]
        assert model.world_size == 1 and model.rank == 0
        assert len(ids) == len(masks) == len(scores)
        locals_ = [(ids, masks, scores)] + [
            (np.empty(0, dtype=np.int64), masks[:0], scores[:0])
            for _ in range(ranks - 1)
        ]
        metadata = dict(
            obj_ids_per_gpu=[item[0] for item in locals_],
            num_obj_per_gpu=[len(item[0]) for item in locals_],
        )
        results = []
        for rank, local in enumerate(locals_):
            calls = []

            def gather(output, value):
                index = len(calls) + 1
                assert index in (1, 2)
                expected = local[index].float().contiguous()
                assert value.dtype == torch.float32 and value.is_contiguous()
                assert torch.equal(value, expected)
                assert len(output) == ranks
                for destination, item in zip(output, locals_):
                    assert destination.shape == item[index].shape
                    destination.copy_(item[index].float())
                calls.append(index)

            globals_ = dict(method.__globals__, dist=types.SimpleNamespace(all_gather=gather))
            source_method = types.FunctionType(
                method.__code__, globals_, method.__name__, method.__defaults__, method.__closure__
            )
            host = types.SimpleNamespace(
                rank=rank, world_size=ranks,
                _propogate_tracker_one_frame_local_gpu=lambda *a, **k: local,
            )
            value = source_method(
                host, inputs["frame_idx"], inputs["num_frames"], inputs["reverse"], [], metadata
            )
            assert calls == [1, 2]
            results.append(value)
        assert all(torch.equal(x, y) for result in results for x, y in zip(result, results[0]))
        records.append(dict(
            frame=int(inputs["frame_idx"]), reverse=bool(inputs["reverse"]),
            objects=len(ids), ranks=ranks, input_dtype=str(masks.dtype),
            output_dtype=str(results[0][0].dtype), source_method_calls=ranks,
            all_gather_calls=ranks * 2,
        ))
        return results[0]

    model.run_tracker_propagation = replay
    return records


def install_partitioned_rank_replay(model, ranks=2):
    """SAM3 full-propagation oracle with real neural rank-local states.

    Backbone/detector execute once. Original per-rank tracker propagation and
    birth/removal execute serially on one GPU. Original rank-zero planning runs
    with world_size=ranks. Its global memory update visits rank-concatenated
    states in one call, retaining global visibility before local memory encoding.
    Only collectives/process dispatch are replaced. Instance-edit routes are not
    covered by this adapter and must not be used with it.
    """
    if ranks < 2 or getattr(model, "is_multiplex", False):
        raise ValueError("partition replay supports SAM3 with at least two ranks")
    collection = model.run_tracker_propagation.__func__
    execution = model.run_tracker_update_execution_phase
    execution_signature = inspect.signature(execution)
    local_propagate = model._propogate_tracker_one_frame_local_gpu
    records = []
    assert model.rank == 0 and model.detector.world_size == 1
    model.world_size = ranks

    def grouped(states):
        output = [[] for _ in range(ranks)]
        for state in states:
            output[state["_reference_logical_rank"]].append(state)
        assert [id(s) for group in output for s in group] == [id(s) for s in states]
        return output

    def propagation(frame_idx, num_frames, reverse, tracker_states_local, tracker_metadata_prev):
        local = [local_propagate(states, frame_idx=frame_idx, reverse=reverse)
                 for states in grouped(tracker_states_local)]
        results = []
        for rank in range(ranks):
            calls = []

            def gather(output, value):
                index = len(calls) + 1
                assert index in (1, 2) and value.dtype == torch.float32 and value.is_contiguous()
                assert torch.equal(value, local[rank][index].float())
                assert len(output) == ranks
                for destination, source in zip(output, local):
                    assert destination.shape == source[index].shape
                    destination.copy_(source[index].float())
                calls.append(index)

            body = types.FunctionType(collection.__code__, dict(collection.__globals__,
                dist=types.SimpleNamespace(all_gather=gather)), collection.__name__,
                collection.__defaults__, collection.__closure__)
            host = types.SimpleNamespace(rank=rank, world_size=ranks,
                _propogate_tracker_one_frame_local_gpu=lambda *a, **k: local[rank])
            results.append(body(host, frame_idx, num_frames, reverse, [], tracker_metadata_prev))
            assert calls == [1, 2]
        assert all(torch.equal(a, b) for result in results for a, b in zip(result, results[0]))
        records.append(dict(phase="propagation",frame=int(frame_idx),reverse=bool(reverse),
            ids_per_rank=[list(map(int, item[0])) for item in local],
            mask_dtypes=[str(item[1].dtype) for item in local],
            score_dtypes=[str(item[2].dtype) for item in local]))
        return results[0]

    def execute(*args, **kwargs):
        bound = execution_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        groups = grouped(arguments["tracker_states_local"])
        result = []
        try:
            for rank, states in enumerate(groups):
                model.rank = rank
                local_arguments = dict(arguments, tracker_states_local=states)
                states = execution(**local_arguments)
                for state in states:
                    state["_reference_logical_rank"] = rank
                result.extend(states)
        finally:
            model.rank = 0
        records.append(dict(phase="execution",frame=int(arguments["frame_idx"]),
            states=[dict(rank=s["_reference_logical_rank"],ids=list(map(int,s["obj_ids"]))) for s in result]))
        return result

    def broadcast(plan, src):
        assert src == 0 and model.rank == 0 and len(plan) == 9
        records.append(dict(phase="plan_broadcast",new_ids=list(map(int,plan[1])),
            new_ranks=list(map(int,plan[2]))))

    model.run_tracker_propagation = propagation
    model.run_tracker_update_execution_phase = execute
    model.broadcast_python_obj_cpu = broadcast
    return records
