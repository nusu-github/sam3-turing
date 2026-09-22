"""Native UTF-8 normalization/tokenization versus the original VE tokenizer."""
import argparse
import hashlib
import html
import json
import os
import random
import subprocess
import time
from pathlib import Path
import ftfy
import regex
import unicodedata
from ftfy import chardata
from sam3.model.tokenizer_ve import SimpleTokenizer


def corpus():
    cases = []
    def add(kind, text): cases.append((kind, text))
    texts = ['', 'truck', 'a yellow butterfly', '赤い車', '車輪と歩行者', 'ＡＢＣ ﾊﾟﾝﾀﾞ',
             'Straße İSTANBUL ΟΣ ΟΣΑ ΟΣ\u0301', 'Привет мир', 'العربية ١٢٣', 'हिन्दी भाषा',
             '👩🏽\u200d💻🏳️\u200d🌈', "we’ll CAN’T he's I'd 'ſ 'll", '<start_of_text> <end_of_text>',
             'hello\x00world\r\nnext\u2028line', '\x1b[31mred\x1b[0m',
             'e\u0301 Å A\u030a ﬃ œ Ｑ', '\ud800 alone \udc00 \ud83d\ude80',
             '&amp;amp; &EACUTE; &SZLIG; &#x1F680; &#0; &#xD800;',
             '<tag> &AMP;\n&EACUTE; <3\n&lt;b&gt;&NTILDE;',
             'voilÃ le travail', 'â€œ like this â€�', 'fÃ cil', 'Ã quele Ã s ',
             'α I’m √±∂†â€™ bad café', 'a '*1000]
    for text in texts: add('text', text)
    for entity in sorted(set(html.entities.html5) | {x[1:] for x in chardata.HTML_ENTITIES}):
        add('entity', 'A &'+entity+' tail')
        add('entity_nested', '&amp;'+entity)
    for value in list(range(256)) + [0xd800,0xdfff,0xfffe,0xffff,0x10ffff,0x110000,99999999]:
        add('numeric_entity', f'&#{value}; &#x{value:x}; x&#{value}z')
    for cp in sorted(set(chardata.WIDTH_MAP) | set(chardata.LIGATURES) | set(range(256))):
        add('character_fix', 'A'+chr(cp)+'x')
    examples = ['café déjà vu', '“hello” — a résumé', 'schön Straße', '日本語の車', 'العربية',
                'Привет', '😀🚀👩🏽', 'İstanbul Ελληνικά', '\x00 null', 'à â ã é ñ ü']
    for text in examples:
        for codec in chardata.CHARMAP_ENCODINGS:
            bad = text.encode('utf8').decode(codec)
            add('mojibake', bad); add('mixed_mojibake', '𐀀 日本 '+bad+' τέλος')
            add('nested_mojibake', bad.encode('utf8').decode(codec))
            add('lossy', text.encode('utf8').decode(codec.replace('sloppy-', ''), errors='replace'))
            add('altered_space', bad.replace('\xa0', ' '))
    rng = random.Random(1901)
    pool = ''.join(examples) + ''.join(chr(i) for i in range(32,256)) + ''.join(chr(i) for i in range(0x2000,0x2080))
    for _ in range(2500): add('random_mixed', ''.join(rng.choices(pool,k=rng.randrange(1,80))))
    for _ in range(1000): add('unicode', ''.join(chr(rng.randrange(0x110000)) for _ in range(rng.randrange(1,25))))
    for cp in range(0x110000):
        c=chr(cp)
        if c.lower()!=c: add('unicode_lower', 'A '+c+' Σ'+c+'Σ')
        if unicodedata.decomposition(c): add('unicode_decomposition', 'x'+c+'y')
        if unicodedata.category(c).startswith('M'): add('combining_sigma', 'ΑΣ'+c+' ΑΣ'+c+'Α')
    # Exercise ftfy's segment length boundary with a token-sized word between
    # whitespace, keeping this a semantic segmentation test rather than BPE stress.
    add('segment_boundary', ' '*999999+'Ã©\n&EACUTE;')
    return cases


def main():
    p=argparse.ArgumentParser();p.add_argument('executable',type=Path)
    p.add_argument('--vocabulary',type=Path,default=Path('sam3/assets/bpe_simple_vocab_16e6.txt.gz'))
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    tokenizer=SimpleTokenizer(a.vocabulary);cases=corpus()
    wire=''.join(text.encode('utf8',errors='surrogatepass').hex()+'\n' for _,text in cases)
    started=time.perf_counter()
    result=subprocess.run([str(a.executable.resolve()),str(a.vocabulary.resolve()),'--hex-stream'],
        input=wire,text=True,capture_output=True,env={**os.environ,'PATH':'/nonexistent'})
    if result.returncode: raise RuntimeError(result.stderr)
    lines=result.stdout.splitlines();assert len(lines)==len(cases),(len(lines),len(cases))
    counts={};failures=[]
    for (kind,text),line in zip(cases,lines):
        cleaned,ids=line.split('\t');actual_clean=bytes.fromhex(cleaned).decode('utf8')
        expected_clean=tokenizer.clean_fn(text);actual_ids=list(map(int,ids.split(',')))
        expected_ids=tokenizer(text,context_length=32)[0].tolist()
        counts[kind]=counts.get(kind,0)+1
        if actual_clean!=expected_clean or actual_ids!=expected_ids:
            failures.append(dict(kind=kind,input=text,expected_clean=expected_clean,actual_clean=actual_clean,
                expected_ids=expected_ids,actual_ids=actual_ids))
    report=dict(cases=len(cases),kinds=counts,failed=len(failures),examples=failures[:20],
                ftfy=ftfy.__version__,unicode=unicodedata.unidata_version,regex=regex.__version__,
                vocabulary_sha256=hashlib.sha256(a.vocabulary.read_bytes()).hexdigest(),
                elapsed_seconds=time.perf_counter()-started,child_path='/nonexistent')
    a.report.write_text(json.dumps(report,ensure_ascii=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'}),flush=True)
    assert not failures, json.dumps(failures[:3],ensure_ascii=True)


if __name__=='__main__':main()
