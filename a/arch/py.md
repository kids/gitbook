# py

线程&进程\
实现上和其他语言差别不大，都是系统调用：\_thread, os.fork()  \
__仅仅差别在GIL限制\_thread在单核上运行

classmethod\&staticmethod\
同样可以作为类的静态函数免实例化食用，区别前者带cls，后者不带\
self\&cls\
当然cls也可以和self一样表示类实例，但一般cls用在以上场景作为类定义

metaclass

```python
class Spam(metaclass=MyMeta, debug=True, synchronize=True):
    pass

class MyMeta(type):
    # Optional
    @classmethod
    def __prepare__(cls, name, bases, *, debug=False, synchronize=False):
        # Custom processing
        pass
        return super().__prepare__(name, bases)

    # Required
    def __new__(cls, name, bases, ns, *, debug=False, synchronize=False):
        # Custom processing
        pass
        return super().__new__(cls, name, bases, ns)

    # Required
    def __init__(self, name, bases, ns, *, debug=False, synchronize=False):
        # Custom processing
        pass
        super().__init__(name, bases, ns)
```

logging\
import logging == import logging; logging.basicConfig() \
所以如果每个类**getLogger**()不同的name相当于各自实例化一个自己的logger

项目架构，抽象类\
\
