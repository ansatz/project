##STRATEGY PATTERN
#var = class variable
#self.var = object instance variable
#http://stackoverflow.com/questions/963965/how-is-this-strategy-pattern-written-in-python-the-sample-in-wikipedia

#--[WHY] rules, not in code!

#--[WHAT] I/O
class base_fnc(object):
    pass
    #def fncExc(self,data1,data2):
    #    return
        #raise NotImplementedError('Exception raised, interface / abstract class!')
    #def nam(self):
    #   return

#--[HOW] DATA <> queue[where,when] .. direct injection objects
class stump( base_fnc ):
    def fncExc(self, d1 , d2):
        return d1 + d2

class MAB(base_fnc ):
    def fncExc(self, d,dd ,acontext ):
        return d + dd + 10

#--[WHO]  API
class context( object ):
    #switch = bsf b #object interface
    def __init__(self, alt_how_class ):
        self.how = alt_how_class

    def exc(self, d, dd ):
        return self. how.fncExc(d,dd, self)

    #def nam(self):
    #    return self.switch.nam()

if __name__ == "__main__":
    #c = context(stump)
    #s=c.exe(1,2) #3
    #sn=c.nam()
    #print s, ' ', sn
    c1 = context(MAB())
    ss=c1.exc(10,3) #13
    print 'yo', c1, ss


#THE SO WAY
#The main differences are :
#
#    You don't need to write any other class nor implementing any interface.
#    Instead you can pass a function reference that will be binded to the method you want.
#    So the functions can still be used stand alone, and the original object can have a default behavior if you want to (the if func == None can be used for that).
#    Indeed, it's clean short and elegant as usual with Python. But you loose
#information : : no explicit interface, so the programmer is assumed as an
#adult knowing what is doing.

#4 ways to dynamically add a method in Python :

#1.  the way I've shown you. But the method will be static, it won't get the "self" argument passed.
#class StrategyExample :
#
#    def __init__(self, func=None) :
#        if func :
#             self.execute = func
#
#    def execute(self) :
#        print "Original execution"
#
#
#def executeReplacement1() :
#        print "Strategy 1"
#
#
#def executeReplacement2() :
#         print "Strategy 2"
#
#if __name__ == "__main__" :
#
#    strat0 = StrategyExample()
#    strat1 = StrategyExample(executeReplacement1)
#    strat2 = StrategyExample(executeReplacement2)
#
#    strat0.execute()
#    strat1.execute()
#    strat2.execute()
##2.  using the class name :
#      StrategyExample.execute = func
#
#3.  all the instance will get "func" as the "execute" method, that will get "self" passed as an argument.
#      binding to an instance only (using types) :
#    strat0.execute = types.MethodType(executeReplacement1, strat0, StrategyExample)
#
#1,3 if need reference to current instance(self) in extended function will get
#Name Error

#proper code is
#import types

#class StrategyExample :
#
#    def __init__(self, func=None) :
#        self.name = "Strategy Example 0"
#        if func :
#****             self.execute = types.MethodType(func, self, StrategyExample)
#
#    def execute(self) :
#        print self.name
#
#
#def executeReplacement1(self) :
#        print self.name + " from execute 1"
#
#
#def executeReplacement2(self) :
#         print self.name + " from execute 2"
#
#if __name__ == "__main__" :
#
#    strat0 = StrategyExample()
#    strat1 = StrategyExample(executeReplacement1)
#    strat1.name = "Strategy Example 1"
#    strat2 = StrategyExample(executeReplacement2)
#    strat2.name = "Strategy Example 2"
#
#    strat0.execute()
#    strat1.execute()
#    strat2.execute()
