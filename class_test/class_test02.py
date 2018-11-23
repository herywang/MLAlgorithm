class Dog:
    def getAge(self):
        return self.__age
    def setAge(self,__new_age):
        self.__age = __new_age
    def msg(self):
        self.__mag()
        print("*"*30)
    def __mag(self):
        print("private msg__________")
d = Dog()

d.setAge(13)
print(d.getAge())
d.msg()
