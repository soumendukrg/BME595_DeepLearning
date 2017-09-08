from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

# Initialize 4 types of Gates
And = AND()
Or = OR()
Not = NOT()
Xor = XOR()

' Test cases for 4 Gates for 1D Input'

# Test cases for AND
print("\nDemonstrating AND Gate functionality using FeedForward Neural Network")
print("And(False, False) = %r" % And(False, False))
print("And(False, True) = %r" % And(False, True))
print("And(True, False) = %r" % And(True, False))
print("And(True, True) = %r" % And(True, True))

# Test cases for OR
print("\nDemonstrating OR Gate functionality using FeedForward Neural Network")
print("Or(False, False) = %r" % Or(False, False))
print("Or(False, True) = %r" % Or(False, True))
print("Or(True, False) = %r" % Or(True, False))
print("Or(True, True) = %r" % Or(True, True))

# Test cases for NOT
print("\nDemonstrating NOT Gate functionality using FeedForward Neural Network")
print("Not(False) = %r" % Not(False))
print("Not(True) = %r" % Not(True))

# Test cases for XOR
print("\nDemonstrating XOR Gate functionality using FeedForward Neural Network")
print("Xor(False, False) = %r" % Xor(False, False))
print("Xor(False, True) = %r" % Xor(False, True))
print("Xor(True, False) = %r" % Xor(True, False))
print("Xor(True, True) = %r" % Xor(True, True))
