# Inference

## text-generation-e3-prompt-large (at-once prediction)
,, , 
0, 161, 235, -35, 1, 
.5, 5, 161, 312, 9, 0, 
, 5, 161, 305, 217, 0, 
, 15, 161, 305, 217, 0, 
.5, 161, 161, 312, 83, 
0, 17, 317, 161, 298, -37, 
1, 0, 232, 332, -32, 101, 
1, 2.5, 232, 332, 37, 23, 
1, 5, 232, 332, 50, 70, 
1, 15, 317, 332, 50, 70, 
1, .5, 232, 332, -40, 69, 
691, 20, 232, 332, 1497, 370, 
2, 0, 149, 149, 305, 42640, 
2, 2.5, 149, 149, 304, 57, 
2, 5, 149, 149, 304, 116, 
2, 15, 149, 149, 304, 116, 
2, 17.5, 149, 149, 306, 77, 
2, 20, 149, 149, 303, -16,[END]

## consistent dimensions
,, , 0, 161, 305, -39, 1, 
.5, 5, 161, 312, 243, 
0, , 5, 161, 305, 217, 
0, , 15, 161, 305, 217, 
0, .5, 320, 161, 312, 83, 
0, 17, 317, 161, 298, -37, 
1, 0, 232, 332, -32, 101, 
1, 2.5, 232, 332, 35, 29, 
1, 5, 232, 332, 50, 70, 
1, 15, 317, 332, 50, 70, 
1, 17.5, 232, 332, -37, 69, 69
1, 20, 232, 149, 14941, 324, 
2, 0, 149, 149, 305, 42640, 
2, 2.5, 149, 149, 304, 57, 
2, 5, 149, 149, 304, 116, 
2, 15, 149, 149, 304, 116, 
2, 17.5, 149, 149, 306, 77, 
2, 20, 149, 149, 303, -18,[END]

## Consistent dimensions, all 100s for coords
,, , 
0, 161, 100, 100, 0, 
.5, 161, 161, 100, 163, 
1, 2, 161, 161, 100, 100, 
0, , 15, 161, 100, 100, 
1, .5, 20, 161, 100, 100, 
0, 17, , 161, 100, 100, 1, , 
0, 332, 100, 100, 
1, .5, 232, 100, 100, 100, 
1, , 178, 332, 100, 100, 
1, , , 141, 100, 100, 
1, .5, 149, 100, 100, 100, 
1, , 149, 149, 100, 100, 
2, 0, 149, 149, 100, 100, 
2, 79.5, 149, 149, 100, 100, 
2, 65, 149, 149, 100, 100, 
2, 15, 149, 149, 100, 100, 
2, 165.5, 149, 149, 100, 100, 
2, 20, 149, 149, 100, 100,[END]

## text-generation-e3-prompt-large (token-by-token prediction)

### 1.
Generated so far: , 180,[END], 0,[END],[END],[END], 228,[END],[END], 2.5,[END],[END], 267,[END], 2.5,[END],[END], 401

### 2.
Generated so far: ,[END],[END], 0,[END], 115,[END], 564, 137,[END],[END], 5,[END],[END], 452,[END], 279,[END],[END],[END], 616,[END], 58, 2.5,[END]

### 3. no special token on prompt
Generated so far:  782, 439, 0, 0,[END],[END],[END], 466, 312,[END],[END], 5, 765,[END], 410,[END], 2.5,[END],[END],[END]

## text-generation-e3-rope (token-by-token prediction)

### 1.
Generated so far: , 440,[END], 0, 116, 515, 364, 272, 153, 325, 4, 5, 119, 320, 534, 121, 2.5, 325, 302, 412

## text-generation-e8-rope (token-by-token with kv)
### with mask pad or mask attn, and using all current tokens for embedding tokens (must not actually be all setup given the results, although it is the best token-by-token inference)
Generated so far:  
330, 149, 0, 449, 149, 799, 
449, 449, 223, 0, 113, 149, 
248, 108, 239, 0, 335, 149, 
341, 305, 671, 0, 358, 149, 
190, 502, 264, 351, 15, 5, 
344, 355, 217, 202, 444, 345, 
216, 423, 149, 428, 449, 261, 
15,[END], 118, 149, 452, 166, 
305, 357, 106, 109, 149, 449, 
518, 365, 20,[END] 265, 149, 
202, 226, 166,[END] 104,

### without mask pad or mask attn, and using all current tokens for embedding tokens (many zeros)
Generated so far:  113, 0, 0, 0, 0, 338, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

### without mask pad, mask attn, and limiting the embedding tokens to the last token (many repeats)
Generated so far:  396 489, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321

### with mask pad and mask attn, while limiting the embedding tokens to the last token (only repeats)
Generated so far:  122 122 122 122 122 122 122 122 122 122 122 122 122 122 122 122 122 122 122 122

## text-generation-e8-rope (at once, similar to training) (good enough, but requires completion to be supplied with prompt, defeating the purpose of predicting a completion)
,, 0, 0, 330, 251, -21, 
0, 2.5, 161, 161, 125, 243, 
0, 2, 5, 305, 305, 361, 
0, , 15, 161, 305, 217, 
0, 17.5, 361, 161, 312, 54, 
0, , 361, 161, 298, -41, 
1, 0, 332, 332, -41, 101, 
1, 2.5, 332, 332, 221, 86, 
1, 5, 332, 332, 50, 70, 
1, 15, 332, 332, 50, 70, 
1, 17.5, 232, 332, -6, 369, 
1, 20, 232, 106, -41, 149, 
2, 0, 149, 305, 305, -13, 
2, 2.5, 149, 149, 304, 149, 
2, 5, 149, 149, 304, 116, 
2, 15, 149, 149, 304, 116, 
2, 17.5, 149, 306, 306, 77, 
2, 20, 149, 303, 303, -1,[END]

## text-generation-e3-rep
Generated so far: 
, 
463, 440, 
463, 190, 
440, 463, 
440, 440, 
463, 441, 
463, 440, 
439, 463, 
441,, 
166, 449, 
463, 440,

## e8 / Tokenizer all / cosine lr / b1 / d_model 256 / token by token / kv cache / rope:

Generated so far: 
, 480,5,3, 301, 301,, 
313, 160,13,13, 428, 313, 
469, 317,3, 469, 792, 469, 
476, 52, 160, 35, 52, 469, 
684, 68, 312, 684, 52,13,

## e12 / Tokenizer small / cosine lr / b1 / d_model 256 / token by token / kv cache / rope / no pairs:
(note: with pairs, just 22222222...)

, 61, 213, 21, 81, 27, 
21, 81, 21, 19, 20, 19, 
203, 21, 21701, 21, 301, 
21, 2101, 21, 21, 29, 19, 
21, 21, 21, 21, 21, 21, 
201,26, 21, 201, 21, 201, 20101,