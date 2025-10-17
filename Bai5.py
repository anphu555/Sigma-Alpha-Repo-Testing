x = int(input("Nhap so thu 1: "))
y = int(input("Nhap so thu 2: "))

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("I'm a student.\n")
    f.write('{:0.5f}\n'.format(1/7))
    f.write(f"{x} + {y} = {x + y}\n")
