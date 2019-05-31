This is the model associated with the paper:

Caplan JS, Williams AH, Marder E (2014) Many parameter sets in a
multicompartment model oscillator are robust to temperature
perturbations J Neurosci. 34(14):496-75

The model code was contributed by Jonathan Caplan.

Usage in a unix/linux environment
---------------------------------

To compile with g++ you can use a command at the shell prompt like:

g++ call_model.cpp model.cpp -lm -O4 -ffast-math -o model.exe

and then to run type:

./model.exe
