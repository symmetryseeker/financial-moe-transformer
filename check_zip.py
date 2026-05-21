import zipfile, os
z = zipfile.ZipFile("C:/Users/lcdell/Desktop/上市企业数据/资产负债表153956739(仅供北京理工大学使用).zip")
for n in z.namelist()[:20]:
    info = z.getinfo(n)
    print(f"  {n} ({info.file_size/1e6:.1f} MB)")
print(f"Total: {len(z.namelist())} files")
