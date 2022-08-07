import io
from pathlib import Path
import os
a= Path('zc.txt')
a.mkdir(parents=True, exist_ok=True)

a_txt ='zc'



file = Path(a,"{}.txt".format(a_txt)).open('w')
Path(a,"aa.txt").open('w')
open('s.txt','w')
#with open('a.txt','w')

Path.resolve(a)
print(Path.resolve(a))
print('s',Path.cwd())


# 拼接出Windows桌面路径
Path(Path.home(), "Desktop")
# 结果： WindowsPath('C:/Users/okmfj/Desktop')

# 拼接出Windows桌面路径
Path.joinpath(Path.home(), "Desktop")
# 结果： WindowsPath('C:/Users/okmfj/Desktop')

# 拼接出当前路径下的“MTool工具”子文件夹路径
Path.cwd() / 'MTool工具'
# 结果： WindowsPath('D:/M工具箱/MTool工具')
print(Path(Path.home(), "Desktop")
,Path.joinpath(Path.home(), "Desktop"),Path.cwd() / 'MTool工具')

a=[0,1,2,3,4,5,6]
b=[5,6,7,8,0,1]

#a[1:6] =[b[i] for i in range(0,5) ]
a[1:6]=b[0:5]
print(a)