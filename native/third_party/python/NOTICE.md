Tokenizer tables and HTML character-reference behavior derive from CPython
3.12 (Python Unicode database 15.0.0), including html.entities.html5, the HTML
invalid-reference/codepoint tables, and Unicode whitespace/word/digit classes.

Source: https://github.com/python/cpython/tree/3.12/Lib/html
License: Python Software Foundation; see LICENSE.txt in this directory.

The tracking session also preserves CPython 3.12 integer-set traversal for
temporary conditioning-frame sets. Probe/growth rules in native/src/frame_order.h
follow Objects/setobject.c; this avoids changing ordered memory attention when
annotations are added at sparse, nonmonotonic frame indices. This is a native
compatibility implementation, not a dependency on the Python interpreter.
Source: https://github.com/python/cpython/blob/v3.12.3/Objects/setobject.c

BPE segmentation classes are frozen from the source environment's regex
2025.11.3 Unicode property results. ICU provides runtime regex matching,
canonical NFC normalization and root-locale lowercasing. ICU/Unicode licensing
must also be included with redistributed ICU binaries/data.
