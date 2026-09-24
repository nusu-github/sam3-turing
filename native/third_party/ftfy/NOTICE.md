The native tokenizer ports the default text-fixing algorithm from ftfy 6.1.1
by Robyn Speer. The generated tables include its badness/UTF-8 detection rules,
character fixes, HTML entity extensions, and sloppy-codec mappings.

Source: https://github.com/rspeer/python-ftfy/tree/v6.1.1
License: MIT; see LICENSE.txt in this directory.

The port implements the tokenizer's default configuration. It does not expose
ftfy's separate optional configuration or explanation APIs.
