# Dùng cách nhập xuất nhị phân, xuất ra file "output.txt" dòng chữ "Hôm nay trời đẹp." sử dụng định dạng UTF-8.
text = "Hôm nay trời đẹp."

fout = open("output.txt", "wb")

fout.write(text.encode('utf-8'))
# byte = bytes(text, "utf-8")
# fout.write(byte)
fout.close()
