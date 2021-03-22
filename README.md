# Python Code Generator
## 1. Introduction
This project aims to generate valid Python code, given an input of the Python function or task description.
For example given input as "Write a python function to add two numbers", a valid function should be generated:

## 2. Dataset
The dataset provided consists of a single text file, containing the Python task description followed by the corresponding Python code.
The task description is styled as a Python comment. There are about 4000 task-function pairs.

Some things to consider:
* The task description is often a short English sentence, however, the corresponding Python code can be of varied lengths
* Indentation is not consistent
  Since the code examples were sourced through many people, some of the code examples have whitespace for indentation while others have tabs.
  Whitespaces also are not always multiples of 2 or 4
* Consistent coding syntax is not followed, for example space around operators 
  Function and variable names are not always snake case, and may have camel case syntax
* There are comments within the code which can cause confusion while splitting into source+target pairs
* Code repetition, but with subtle differences
  For example, the code to find the factorial has around 10 different implementations
* Presence of test inputs such as numbers and random strings
The original dataset can be found here: https://drive.google.com/file/d/1rHb0FQ5z5ZpaY2HpyFGY6CeyDG0kTLoO/view

### 2.1 Data Cleaning
Autopep8 is a library that helps to format Python code as per Pep8 coding standards.
It takes care of some very obvious coding style fixes, such as:
* Indentation (indentation as whitespaces, multiples of four, hanging indents, trailing whitespace, etc)
  This helps in creating having a standard representation of indents while training.
* Adding whitespace around keywords
  This helps with tokenizing the keywords and operators correctly during data preparation
* Adding whitespaces around operators
  For example:
  if a>b:
  gets converted to
  if a > b:

Autopep8 handles most of the data cleaning and code standardizing problems. However there are still comments left within the code, which had to be removed manually in many places. There was also a problem where Autopep8 did not remove or convert all the tabs correctly into whitespaces. However the model was able to learn the indentation quite well.

### 2.2 Data Preparation
For the purposes of training, only code examples within 300 characters were considered.
This was to reduce the difference in the lengths of the source and target tokens.

**Tokenizers**
* Spacy tokenizer was used as is since it gave good results
  Spacy tokenizes punctuations quite well, so operators like ':', '(' etc are treated as separate tokens, which is what we require
* Spacy also considers newlines as a single token, so we get '\n'. Indentation tokens are of the form '\n ', '\n ' etc, which is also good
* There were a few operands, however, that weren't getting tokenized correctly. So we need to add a few regexes to add spaces.
  For example: [1, 2, 3] gets tokenized as '[1', ',', '2', ',', '3]'
  Adding a regex to convert it to
  [ 1, 2, 3 ] , gives tokens of the form '[', '1', ',', '2', ',', '3', ']'

Example input:
def add_two_numbers ( num1 , num2 ): \n sum = num1 + num2 \n return sum
Tokenized output:
['def', 'add_two_numbers', '(', 'num1', ',', 'num2', '):', '\n ', 'sum', '=', 'num1', '+', 'num2', '\n ', 'return', 'sum']

While building the vocabularies, we normally keep a minimum frequency count of 2. However the code generated with this vocab setup was not generating the function name. Hence, min frequency count of 1 has been used for target vocabulary.
Source vocabulary length: 1195
Target vocabulary length: 4412

The actual core python specific tokens are not many, however the target vocabulary is larger because of a huge variety in variable names, function names, test inputs used, etc.

**Embeddings**
The initial model did not use custom embeddings, so the output code generated, though syntactically decent, was not always matching with the source question asked. Also the code was not always of the right syntax. In order to fix this, custom embeddings were trained.

Gensim is a Python library that can be used to train custom embeddings. Given a corpus of tokens, it generate Word2Vec vectors for these tokens. These vectors can then be set into the source and target vocabularies generated previously.

The first test run had separate embeddings trained for the source and target, and while it was better than the model without custom embeddings, the model gave better results when both the source and target tokens were added together to form a corpus. This might be because it provided better conceptual relationships between the source and target tokens.

Once the custom embeddings are trained on (source + target) tokens, they have to be loaded into the source and target vocabularies, only for the tokens present in these vocabularies. Hence the size of the source and target vocabularies do not increase, but the tokens within them get loaded with vectors that are contextually related.

## Model
The model used is Seq2Seq with Multi-Head attention. This is the same model what was used in the previous sessions. The only difference is in increasing the max length at the decoder so that it can return longer sentences (ie python code). Also the embedding layers were loaded with pretrained embedding vectors for better accuracy.

The model was heavily overfitting right from the beginning, so only two Encoder and Decoder layers were used, to reduce model parameters. Also a higher dropout rate had to be configured.
More number of heads helped in finetuning the Final Model Config parameters:

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 768 
ENC_LAYERS = 2
DEC_LAYERS = 2 
ENC_HEADS = 4
DEC_HEADS = 4
ENC_PF_DIM = 1024
DEC_PF_DIM = 1024
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
LEARNING_RATE = 0.0005
Model Parameters: 24,149,820 trainable parameters

## Loss Function
The loss function initially used was CrossEntropy Loss. This loss function is commonly used in NLP models, but it was causing my model to heavily overfit. The model's training and validation loss were both high, wnd while they did decrease during training, the final validation loss was always higher than the training loss. Higher dropout, lesser model parameters, and better embedding layers did bring down this loss quite a bit but it wasn't sufficient.
Label smoothing is an option that can be used to make the model less overconfident. It is frequently used with Cross Entropy and it replaces one-hot encoded label vector y_hot with a mixture of y_hot and the uniform distribution.

Using label smoothing, the difference between the training and validation loss was reduced significantly, and the model began to output much better results during inference.

## Examples
Examples are present in the colab file along with the attention graphs.

## Notes
* Certain operands, like '==', get split into '=', '=' by spacy. This causes the code generated to look like a = = b.
  While 'a < = b' is syntactically correct, 'a = = b is not. spacy provides a way to define custom token roles for handling these kinds of cases, but I did not get time to implement it.
* No data augmentation techniques were used. Since the model was overfitting, these could have helped.
* Data cleaning and preparation took the most time and were also the most important. Small things like spaces around operands proved to have a large impact on the quality of the code generated.
