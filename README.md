# HangmanChallenge

A small library that plays the game of hangman.<br />

Steps to play a game - see HangmanChallenge/jupyter/demo.ipynb for a run through:<br />
Run tests /HangmanChallenge/tests/pytest<br />
<br />
- Load a dictionary of words (HangmanChallenge/data/words_250000_train.txt)<br />
- Create an instance of HangmanChallenge.hangman.core.api.API passing it the dictionary.  This object will then select a word at random, and then return its masked view i.e. are -> ___.  The aim is to keep trying to guess letters by calling API().guess().  The api will allow you 6 lives before the game is lost.<br />
- Select a player object to use from HangmanChallenge.hangman.model that adheres to the player.IPlayer interface (see below for different implementations).<br />
- Use HangmanChallenge.hangman.core.api.API<br />
<br />
<br />
There are two main approaches to selecting letter guesses are:<br />
<br />
- A heuristic approach that selects letters based on what appears most frequently in the input dictionary.  First the dictionary is reduced to leave words with the same length.  Then each incorrect guess filters out all words from the dictionary that contain that letter.  For correct guesses words that do not contain that letter(s) in the same positions are also removed.<br />
<br />
    player_heuristic = hangman.model.basic.Heuristic(words)<br />
<br />
- There is a neural network approach that uses a combination of the heuristic approach and a LSTM model.  For training we take the input training dictionary of words and create all combinations of each word replacing 1 to n-2 characters with an underscore (to simulate different hangman game states).  All words (including newly added masked combinations) are split into ngrams of lengths ranging between 2 and 7 characters in length.  Each ngram is split into two; x=[:-1] and y[-1] with the idea that we will use a sequence of x to predict one character y.  We also apply the same logic to each ngram in reverse.  This is the data that is then used to train an LSTM keras model.  We experimented with both a dual layer bidirectional model and a tri layer model - please see HangmanChallenge.hangman.model.ml.config for exact specifications.  The heuristic approach is used until at least 50% of the letters have been guessed then we take an intersection of the top 3 heuristic guesses with the ML generated guesses.<br />
<br />
    model = hangman.model.ml.LSTModel("load_model_weights", config=TriLayer(), pad_sequence=False)<br />
    player_lstm = hangman.model.ml.NNPlayer(words, model=model, verbose=False, heuristic_thershold=0.5)<br />