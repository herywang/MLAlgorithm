
class Kls(object):
    no_inst = 0
    def __init__(self):
        Kls.no_inst = Kls.no_inst + 1
        print("class" + str(Kls.no_inst))
    @classmethod
    def get_no_of_instance(cls_obj):
        return cls_obj.no_inst

ik1 = Kls()
print(ik1.no_inst)
ik2 = Kls()
print(ik2.no_inst)
print(ik1 == ik2)
print(ik1.get_no_of_instance())
print(ik2.get_no_of_instance())
