# f = open("a.txt")
# print(f.read())
# f.close()

# with open("a.txt") as f:
#     print(f.read())
    
fName = str(input("Nhap ten file de mo: "))

try:
    with open (fName) as f:
        print(f.read())
except FileNotFoundError:
    print("Loi: File khong ton tai")

    
