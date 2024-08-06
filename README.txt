In the file SURP_functions.py, you can find all of the functions used to generate the various results we found.
The most important of these are any of the later functions that begin with output - these output an HTML file that contains 
information on the topic in the name of the function.

Also of particular interest are the create_dp_table functions, which implement a dynamic programming formula (iteratively)
that can then be used to find the most optimal sentence alignment for each pair of texts.

"google.1grams.txt", "normal.11.counts.txt", and "simple.11.counts.txt" are .txt files containing a list of all words that 
occurred in each of a Google dataset, Normal Wikipedia dataset, and Simple Wikipedia dataset.

In the Summary Spreadsheets folder you can find all the key results using these functions.

A collection of the most useful result files can be found in sentence_alignment/Most Useful.