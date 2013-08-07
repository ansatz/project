##############################################

https://bitbucket.org/BruceEckel/python-3-patterns-idioms/src/63059be05961b2b28fa0340898db2941ae1eefc4/src/Visitor.rst?at=default
#VST , double dispatch 
#1.used when have hierachical base class that you cant touch
#2. allows you to extend primary type by creating a separate class hiearachy, and using dynamic lazy eval
#3. allows for an iterator, such as tree-traversal, AST
#4. allows traversal of the CMD [where] structure
#base class, cannot change
class mysqlFixedData(object):
	def accept(self, visitor):
		visitor.join(self)
	def select(self, ponyORM):
	def parseRoute(self, ponyORM):
	def  ponyORM(self):


#~nodes [where,when] 
class static(mysqlFD): pass
class sequential(mysqlFD): pass
class indicator(mysqlFD): pass
#iterator:
def fusionTree(n):
	dbs = mysqlFD.__subclasses__()
	for i in range(n):
		yield.random.choice(dbs)()

#interface to extend base
class Visitor:
	def __str__(self):
		return self.__class__.__name__
class Join(Visitor): pass
class fastSmallSet(Join): pass
class dissected(Join): pass

#extended methods
class rubix(dissected):
	def join(self, mysqlFD):
		mysqlFD.parseRoute(self)
class alerts(fastSmallSet):
	def join(self, mysqlFD):
		mysqlFD.select(self)


rbx = rubix()
alrt = alerts()
for data in fusionTree(7):
	mysqlFixedData.accept(rbx)
	mysqlFixedData.accept(alrt)


###############################################
#CMD parameterless call-back operation
#invoker { [WHERE]space(elevator, p-queue, fusion-tree),
#	   [WHEN] time(events(window-size, undo) )        --'obsvr,we'll call you'--
#	 }
#receiver ([WHO]-api, setter)                             --WHAT,WHY--
#client (interface-cmd-api)

class


###############################################


