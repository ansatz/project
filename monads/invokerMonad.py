import strategy as api
from monad_decorators import *
from invokerMonad import * as ivkMnd

#@jordie decorator to traversal-invoke fnc{elevator,priority,deque}
def jordie(traversal):
	def wrapped(): #ivkMnd.parser():
		return traversal()
	return wrapped

#@whorf decorator to honor-invoke fnc{sequence}
def whorf(honor):	
	def wrapped(): #ivkMnd.response():
		return honor()
	return wrapped #ivkMnd.response
#q is asynchronous

#@ensen is decorator to event-functions{server-maintain}
def ensen(event)
	def wrapped(): #ivkMnd.observer()
		return event()
	return wrapped #ivkMnd.observer

#@Data decorator to query-functions
def Data(query): 
	def wrapped: #ivkMnd.visitor():
		return query()
	return wrapped #ivkMnd.visitor

####################################################################
class invokerMonad(api):
	@whorf
	def response(self, event):
		return print event


	@jordie
	def conduits(self, space='elevator'):
		return print space

	@ensen
	def observer(self, eventList)
		eventList = [e(context()) for e,context in eventList] 

	@Data
	def visitor(self, query)
		return ponyORM(query)

###################################################################
#Rx programming as creating events as streams >> valued lessons


