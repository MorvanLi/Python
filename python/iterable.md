## 迭代器和可迭代对象

在python中实现了__iter__方法的对象是可迭代的，实现了next()方法的对象是迭代器，这样说起来有点拗口，实际上要想让一个迭代器工作，至少要实现__iter__方法和next方法



```python
>>> from collections.abc import Iterable
>>> a = 1234
>>> isinstance(a, Iterable)
False
>>> b = [1, 2, 3, 4]
>>> isinstance(b, Iterable)
True

```



***

那么列表是否是一个迭代器呢？

```python
>>> from collections.abc import Iterator
>>> isinstance(b, Iterator)
False
>>> b = iter(b)
>>> isinstance(b, Iterator)
True

```





自定义一个迭代器

```python
class MyIterator:
    def __init__(self, data):
        self.data = data
        self.idx = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == 0:
            raise StopIteration
        self.idx = self.idx - 1
        return self.data[self.idx]


myiterator = MyIterator("1234")
print(next(myiterator))
print(next(myiterator))
print(next(myiterator))
print(next(myiterator))


>>>4
>>>3
>>>2
>>>1
```

[参考链接]([1.11 Python 中的可迭代对象、迭代器与生成器 — 可乐python说 1.0.0 文档 (kelepython.readthedocs.io)](https://kelepython.readthedocs.io/zh/latest/c01/c01_11.html))
