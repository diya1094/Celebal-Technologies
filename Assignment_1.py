def lowerTri(n):
    for i in range(n,0,-1):
        print("*"*i)

def upperTri(n):
    for i in range(1, n + 1):
        print("*" * i)

def pyramid(n):
    for i in range(n):
        s = ""
        for space in range(n - i - 1):
            s += " "
        for star in range(2 * i + 1):
            s += "*"
        print(s)

row = int(input("Enter number of rows: "))

print("Lower Triangle:")
lowerTri(row)
print("Upper Triangle:")    
upperTri(row)
print("Pyramid:")
pyramid(row)