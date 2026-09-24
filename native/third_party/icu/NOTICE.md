ICU 74.2 is the validated native Unicode dependency. The tokenizer uses its
Unicode 15.1 NFC/lowercase data and regex engine; character classification for
BPE and Python regex behavior is frozen separately in generated tables.

Source: https://github.com/unicode-org/icu/tree/release-74-2
Build guidance (Windows/Linux): https://unicode-org.github.io/icu/userguide/icu4c/build.html
License: see LICENSE.txt. Include this license when redistributing ICU code/data.
