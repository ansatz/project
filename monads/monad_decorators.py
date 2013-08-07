#############################################################3
#A decorator is a function (such as foobar in the above example) that takes a function object as an argument, and returns a function object as a return value. 
def makebold(fn):
	def wrapped():
		return "<b>" + fn() + "</b>"
	return wrapped

def makeitalic(fn):
	def wrapped():
		return "<i>" + fn() + "</i>"
	return wrapped

@makebold
@makeitalic
def hello():
	return "hello world"

print hello() ## returns <b><i>hello world</i></b
#############################################################3
##
#MONAD(adjective): not a thing, is a typeclass, not a type--container because instance because of type
#rED box can store not because Red because its box
#Red paper not a container
##
#function: val to val a->b
#functor: can be container(~list)/computation [a]->b
#monad: valA to context of b a->f->[b] {1:'a'} 1->string... programmable (:)
##
#1.fnc->stations
#2.inputs->astronauts
#3.all output in suit(monad-value)
#
#arrow: a->[f]->b
#
###############################################################
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


