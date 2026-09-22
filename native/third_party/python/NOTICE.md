Tokenizer tables and HTML character-reference behavior derive from CPython
3.12 (Python Unicode database 15.0.0), including html.entities.html5, the HTML
invalid-reference/codepoint tables, and Unicode whitespace/word/digit classes.

Source: https://github.com/python/cpython/tree/3.12/Lib/html
License: Python Software Foundation; see LICENSE.txt in this directory.

BPE segmentation classes are frozen from the source environment's regex
2025.11.3 Unicode property results. ICU provides runtime regex matching,
canonical NFC normalization and root-locale lowercasing. ICU/Unicode licensing
must also be included with redistributed ICU binaries/data.
