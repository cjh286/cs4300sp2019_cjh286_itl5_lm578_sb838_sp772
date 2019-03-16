# Notes From the Set Up Meeting

If your dataset is a csv - you have some kind of metadata representation,
load all csv rows into a database, and keep a lightweight thing in the memory
like a term doc matrix, and you can get the index of some row
once you know index of row, you can get the dataset

Keep raw docs in database, use lightweight operations for docid, then lookup doc_id
in database