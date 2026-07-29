from pymerkle import InmemoryTree
import math

# Create an empty tree
tree = InmemoryTree()

def getHeight(nLeaf):
    return math.ceil(math.log2(nLeaf)) if nLeaf>0 else 0

initial_hash=tree.get_state()

print("===== EMPTY TREE =====")
print("Root Hash :", tree.get_state().hex())
print("Size      :", tree.get_size())
print("Leaves    :", len(tree.leaves))
print("Height    :", getHeight(len(tree.leaves)) )

# Add some records
tree.append_entry(b"Apple")
tree.append_entry(b"Banana")
tree.append_entry(b"Orange")
tree.append_entry(b"Grapes")

print("\n===== AFTER ADDING RECORDS =====")
print("Root Hash :", tree.get_state().hex())
print("Size      :", tree.get_size())
print("Leaves    :", len(tree.leaves))
print("Height    :", getHeight(len(tree.leaves)) )

# using the hashlib
import hashlib


data1=b"Apple"
data2=b"Banana"
data3=b"Orange"
data4=b"Grapes"

print("Data:Apple,Banana, Orange, Grapes")

# Individual Hash
hash1=hashlib.sha256(data1).digest()
hash2=hashlib.sha256(data2).digest()
hash3=hashlib.sha256(data3).digest()
hash4=hashlib.sha256(data4).digest()

# Hash combine level 1
hash12=hashlib.sha256(hash1+hash2).digest()
hash34=hashlib.sha256(hash3+hash4).digest()

# Root hash
parent=hashlib.sha256(hash12+hash34).digest()
main=hashlib.sha256(parent+initial_hash).digest()
print(
f'''
    
                                                                                                                    |----hash1({hash1.hex()})
                                                                                                                    |
                                                |------hash12({hash12.hex()}) 
                                                |                                                                   |
                                                                                                                    |----hash2({hash2.hex()})
parent({main.hex()})
                                                                                                                    |----hash1({hash3.hex()})
                                                |                                                                   |
                                                |------hash34({hash34.hex()})
                                                                                                                    |
                                                                                                                    |----hash1({hash4.hex()})



''')