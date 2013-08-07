#TITLE: python design patterns
#Hello,
#
#I am trying to stack the four patterns as follow: [STRATEGY,COMMANDER,OBSERVER,VISITOR]
#I would like advice on light-weight, pythonic design.
#
#STRATEGY: Conceptually, here I define what needs to be done.  I then extend how to accomplish that; and finally create an interface.
#[WHAT]
class threshold(object):
    def indicator(self, entropy):
		return fpCrv
    def label(self):    
		return rnkCI
#[HOW]
class bayesBn( threshold ):
    def indicator(self, ent, ,size, aMiner ):
        kernel = stats.gaussian_kde(ent)
		...		
        return fpCurve

    def label( self, entr):
        return rankedCI
		
class hashKrnl( threshold ):
    def indicator(self, ent, size, aMiner):
        return fpCurve
#[WHO] api <receiver>
CONTEXT={'bb0':'bayesBn',
	  'hh1':'hashKrnl'}
class dmAPI(object):
    def __init__(self,contextSwitches ):
        self.how = contextSwitches

    def fpCurve(self, ent, size ):
        return self.how.indicator(ent, size, self)

	def rankedCI(self, H_I):
        return self.how.label(H_I, self)	

#<client>		
import ponyORM
import fuquit as srvr
EVENTS = []
class cmdr(dmAPI,srvr,ponyORM): #client
	#client(noun)	
	def __init_ _(self, space, time, data ):
		self.space = space #traversal{elevator,priority}
		self.time = time #events
		self.data = data #ponyORM
	def __repr__(self):
		return "object set as" + 
	#invokers
	def jordie(self): #traversal -generator recursive not pass function object, maintain context
		return datastruct
	def ensen(self, listObsv): #observer
		#generator
		SRV.ON and yield
		or return print"hailing-frequencies-clear"
	def whorf(self):
		pass #respond to events

	def theQ(self,pony): #visitor
		pass #query data

	def tobridge(self):
		#start server fuquit
		
		#event c all-back (obsrv ptrn ? continuation monad, promises are functional, cb are imperative)
		cntx =  [e for CONTEXT.get( urlparse(event) )
		
		#set crew-member;
		crews = self.dmAPI( cntx() ) for i in  
		crew= lambda cntx: self.dmAPI( cntx() )

		#query data -visitor
		self.accept()	
	
	def raiseShield(self,EVENTS):
		#while-block
		lambda: (!EVENTS and return 1) \
				or return 0
				
	def acceptQ(self, visitorQ):
		visitorQ.pony(self)
		
	def engage(self): #invoker
		
	def setObj(self, cntx): #receiver
		return lambda cntx: self.dmAPI( cntx() )
	def traverse(self, dataStruct): #[stack-elevator, priority-queue]
		pass	
	@traversal
	def dataStruct(cmdr):
		

#[HOW] DATA <> queue[where,when] .. direct injection objects
#[WHO]- 


if __name__ == "__main__":
    c1 = dmAPI( bayesBn() )
    ss=c1.exc(10,3) #13
    print 'yo', c1, ss
	#tng quotes
	CMDR=cmdr.tobridge(holodeck=) #set objects,traversal
	CMDR.engage        #respond to events, query data, return result
