from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

# Initialize 4 types of Gates
And = AND()
And.train()
Or = OR()
Or.train()
Not = NOT()
Not.train()
Xor = XOR()
Xor.train()

' Test cases for 4 Gates for 1D Input'

# Test cases for AND
print("\nDemonstrating AND Gate functionality using FeedForward Neural Network")
print("And(False, False) = %r" % And.forward(False, False))
print("And(False, True) = %r" % And.forward(False, True))
print("And(True, False) = %r" % And.forward(True, False))
print("And(True, True) = %r" % And.forward(True, True))

# Test cases for OR
print("\nDemonstrating OR Gate functionality using FeedForward Neural Network")
print("Or(False, False) = %r" % Or.forward(False, False))
print("Or(False, True) = %r" % Or.forward(False, True))
print("Or(True, False) = %r" % Or.forward(True, False))
print("Or(True, True) = %r" % Or.forward(True, True))

# Test cases for NOT
print("\nDemonstrating NOT Gate functionality using FeedForward Neural Network")
print("Not(False) = %r" % Not.forward(False))
print("Not(True) = %r" % Not.forward(True))

# Test cases for XOR
print("\nDemonstrating XOR Gate functionality using FeedForward Neural Network")
print("Xor(False, False) = %r" % Xor.forward(False, False))
print("Xor(False, True) = %r" % Xor.forward(False, True))
print("Xor(True, False) = %r" % Xor.forward(True, False))
print("Xor(True, True) = %r" % Xor.forward(True, True))


