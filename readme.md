# License plate processing

In this project, we wrote a license plate recognition program with artificial intelligence

## Modules

```python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget
```

## Usage

This code opens an image in Python and uses the OpenCV library.

```python
im = cv2.imread('2/22.jpg')
```

Opens the image in a window, waits for the user to press a key, and then closes the window.

```python
cv2.imshow('ax',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This code will open the list of all files and folders in directory "1" and return them as a list.

```python
xfiles = os.listdir('1')
```

show file

```python
print(files)
```

This code resizes the image named "im" inside the variable using the resize function in OpenCV to new dimensions (8x32) and places it inside the "im2" variable.

```python
im2 = cv2.resize(im,(8,32))
im2.shape
```

Change the color of the photo

```python
im3 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
im3.shape
print(im3)
```

This code reorders the image stored in the "im3" variable into a new shape that has dimensions of 256 and places it in the "im4" variable.

```python
im4 = im3.reshape(256)
print(im4)
im4.shape
```

This code first reads the list of all files and folders in directory "1". It then creates an empty matrix named "x" with dimensions (number of files, 256). It also creates an empty array called "y". Then, for each file in the "1" directory, it reads the corresponding image and first resizes it to (8x32), then converts it to gray color space, and finally reorders it into a one-dimensional vector with dimensions of 256 and in the matrix Saves "x". It also adds the label "1" to the array "y".

```python
files = os.listdir('1')

x = np.empty((0,256))

y = np.array([])

for file_name in files:
    im = cv2.imread('1/' + file_name)
    im2 = cv2.resize(im,(8,32))
    im3 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    im4 = im3.reshape(256)
    x=np.append(x,[im4],axis=0)
    y=np.append(y,'1')
```

This code creates an empty matrix named "x" with dimensions (0, 192) and an empty array named "y". Then, for each number in the range of numbers 1 to 9, it reads the list of all files and folders in the directory corresponding to that number and reads the corresponding image for each file. Then it resizes the image to (8x24) and converts it to a gray color space and converts it to a binary image using the thresholding method. Finally, it reorders the image into a one-dimensional vector with dimensions of 192 and stores it in the "x" matrix. It also adds the label corresponding to the file directory to the "y" array.

```python
x = np.empty((0,192))
y = np.array([])

for k in range(1,10):
    files = os.listdir(str(k))
    for file_name in files:
        im = cv2.imread(str(k) + '/' + file_name)
        im2 = cv2.resize(im,(8,24))
        im3 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
        ret,im3 = cv2.threshold(im3,127,255,cv2.THRESH_BINARY)
        im4 = im3.flatten()
        x=np.append(x,[im4],axis=0)
        y=np.append(y,k)

y.shape
x.shape
```

<h1> Building a model to recognize numbers </h1>

```python
from sklearn import linear_model
from sklearn.model_selection import train_test_split
```

Dividing the input data into two sets of training and testing with a ratio of 80% and 20% respectively.

```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
out = model.predict(X_test)
out
y_test
```

<h1> The second session of license plate processing </h1>

We use to remove threshold noise, we read a photo of a license plate for processing and from

```python
img_pelak = cv2.imread('pelak.png')
img_pelak = cv2.cvtColor(img_pelak,cv2.COLOR_BGR2GRAY)
ret,img_pelak = cv2.threshold(img_pelak,127,255,cv2.THRESH_BINARY)
```

We turn the photo into gray

```python
img_pelak0 = cv2.imread('pelak.png',0)
```

show image

```python
cv2.imshow('ax',img_pelak0)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_pelak0.shape
```

In order to draw a line between the license plate numbers, we use the following formula

```python
s = 90 - np.sum(img_pelak0,axis=0,keepdims=True)/255
print(s[0])
plt.close()
plt.plot(s[0])
plt.show()
```

We draw a line between the places where there is a white license plate in the photo

```python
pelak = img_pelak0.copy()

pelak = cv2.line(pelak,(160,0),(160,90),(0,0,0),1)

cv2.imshow('ax',pelak)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```python
img_pelak = cv2.imread('pelak.png')
ret,img_pelak = cv2.threshold(img_pelak,127,255,cv2.THRESH_BINARY)

pelak = img_pelak0.copy()
xi = 0
xi1 = 0
xi2 = 0


flag1 = False
flag2 = False


for i in s[0]:
    xi += 1

    if i > 9 and flag1==True and flag2 == False:
        
        flag2 = True

    if i < 5:
        cv2.line(pelak,(xi,0),(xi,90),(0,0,0),1)
        
        if flag1 == False:
            xi1 = xi
        
        if flag2 == True:
            xi2 = xi
            flag2 = False
            flag1 = False
            img1 = img_pelak[:,xi1:xi2]
            
            x1 = np.empty((0,192))
            im2 = cv2.resize(img1,(8,24))
            im3 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
            im4 = im3.flatten()
            x1 = np.append(x1,[im4],axis=0)
            natije = model.predict(x1)
            print(natije)
            
            
            cv2.imshow('ax',img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            xi1 = xi2

        flag1 = True
```

## Result

This project was written by Majid Tajanjari and the Aiolearn team, and we need your support!❤️

# پردازش پلاک

در این پروژه یک برنامه تشخیص پلاک با هوش مصنوعی نوشتیم

## ماژول

```python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget
```

## نحوه استفاده

این کد یک تصویر را در پایتون باز می کند و از کتابخانه OpenCV استفاده می کند.

```python
im = cv2.imread('2/22.jpg')
```

تصویر را در یک پنجره باز می کند، منتظر می ماند تا کاربر کلیدی را فشار دهد و سپس پنجره را می بندد.

```python
cv2.imshow('ax',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

این کد لیست تمام فایل ها و پوشه ها را در دایرکتوری "1" باز می کند و آنها را به صورت لیست برمی گرداند.

```python
xfiles = os.listdir('1')
```

نمایش فایل

```python
print(files)
```

این کد با استفاده از تابع resize در OpenCV اندازه تصویر با نام "im" را در داخل متغیر به ابعاد جدید (8x32) تغییر می دهد و آن را در داخل متغیر "im2" قرار می دهد.

```python
im2 = cv2.resize(im,(8,32))
im2.shape
```

رنگ عکس را تغییر دهید

```python
im3 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
im3.shape
print(im3)
```

این کد، تصویر ذخیره شده در متغیر "im3" را به شکل جدیدی با ابعاد 256 تغییر می دهد و آن را در متغیر "im4" قرار می دهد.

```python
im4 = im3.reshape(256)
print(im4)
im4.shape
```

این کد ابتدا لیست تمام فایل ها و پوشه های دایرکتوری "1" را می خواند. سپس یک ماتریس خالی به نام "x" با ابعاد (تعداد فایل، 256) ایجاد می کند. همچنین یک آرایه خالی به نام "y" ایجاد می کند. سپس برای هر فایلی که در دایرکتوری "1" قرار دارد، تصویر مربوطه را می خواند و ابتدا اندازه آن را به (8x32) تغییر اندازه می دهد، سپس آن را به فضای رنگی خاکستری تبدیل می کند و در نهایت آن را به یک وکتور یک بعدی با ابعاد 256 و 256 تغییر می دهد. ماتریس "x" را ذخیره می کند. همچنین برچسب "1" را به آرایه "y" اضافه می کند.

```python
files = os.listdir('1')

x = np.empty((0,256))

y = np.array([])

for file_name in files:
    im = cv2.imread('1/' + file_name)
    im2 = cv2.resize(im,(8,32))
    im3 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    im4 = im3.reshape(256)
    x=np.append(x,[im4],axis=0)
    y=np.append(y,'1')
```

این کد یک ماتریس خالی به نام "x" با ابعاد (0، 192) و یک آرایه خالی به نام "y" ایجاد می کند. سپس به ازای هر عدد در محدوده اعداد 1 تا 9، لیست تمامی فایل ها و پوشه های موجود در دایرکتوری مربوط به آن عدد را می خواند و تصویر مربوط به هر فایل را می خواند. سپس اندازه تصویر را به (8x24) تغییر می دهد و آن را به یک فضای رنگی خاکستری تبدیل می کند و با استفاده از روش آستانه گذاری آن را به یک تصویر باینری تبدیل می کند. در نهایت، تصویر را به یک بردار یک بعدی با ابعاد 192 مرتب می کند و آن را در ماتریس "x" ذخیره می کند. همچنین برچسب مربوط به فهرست فایل را به آرایه "y" اضافه می کند.

```python
x = np.empty((0,192))
y = np.array([])

for k in range(1,10):
    files = os.listdir(str(k))
    for file_name in files:
        im = cv2.imread(str(k) + '/' + file_name)
        im2 = cv2.resize(im,(8,24))
        im3 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
        ret,im3 = cv2.threshold(im3,127,255,cv2.THRESH_BINARY)
        im4 = im3.flatten()
        x=np.append(x,[im4],axis=0)
        y=np.append(y,k)

y.shape
x.shape
```

<h1> ساخت مدلی برای تشخیص اعداد </h1>

```python
from sklearn import linear_model
from sklearn.model_selection import train_test_split
```

تقسیم داده های ورودی به دو مجموعه آموزش و تست به ترتیب با نسبت 80% و 20%.

```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
out = model.predict(X_test)
out
y_test
```

<h1> جلسه دوم پردازش پلاک </h1>

ما برای حذف نویز آستانه استفاده می کنیم، عکس پلاک خودرو را برای پردازش و از آن می خوانیم

```python
img_pelak = cv2.imread('pelak.png')
img_pelak = cv2.cvtColor(img_pelak,cv2.COLOR_BGR2GRAY)
ret,img_pelak = cv2.threshold(img_pelak,127,255,cv2.THRESH_BINARY)
```

عکس را خاکستری می کنیم

```python
img_pelak0 = cv2.imread('pelak.png',0)
```

نشان دادن تصویر

```python
cv2.imshow('ax',img_pelak0)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_pelak0.shape
```

برای اینکه بین شماره پلاک ها خط بکشیم از فرمول زیر استفاده می کنیم

```python
s = 90 - np.sum(img_pelak0,axis=0,keepdims=True)/255
print(s[0])
plt.close()
plt.plot(s[0])
plt.show()
```

بین جاهایی که در عکس پلاک سفید وجود دارد خط می کشیم

```python
pelak = img_pelak0.copy()

pelak = cv2.line(pelak,(160,0),(160,90),(0,0,0),1)

cv2.imshow('ax',pelak)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```python
img_pelak = cv2.imread('pelak.png')
ret,img_pelak = cv2.threshold(img_pelak,127,255,cv2.THRESH_BINARY)

pelak = img_pelak0.copy()
xi = 0
xi1 = 0
xi2 = 0


flag1 = False
flag2 = False


for i in s[0]:
    xi += 1

    if i > 9 and flag1==True and flag2 == False:
        
        flag2 = True

    if i < 5:
        cv2.line(pelak,(xi,0),(xi,90),(0,0,0),1)
        
        if flag1 == False:
            xi1 = xi
        
        if flag2 == True:
            xi2 = xi
            flag2 = False
            flag1 = False
            img1 = img_pelak[:,xi1:xi2]
            
            x1 = np.empty((0,192))
            im2 = cv2.resize(img1,(8,24))
            im3 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
            im4 = im3.flatten()
            x1 = np.append(x1,[im4],axis=0)
            natije = model.predict(x1)
            print(natije)
            
            
            cv2.imshow('ax',img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            xi1 = xi2

        flag1 = True
```

## نتیجه

این پروژه توسط مجید تجن جاری و تیم Aiolearn نوشته شده است و ما به حمایت شما نیازمندیم!❤️