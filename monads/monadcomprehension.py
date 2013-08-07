#monad {1.computation-builder , 2.container}

#monad comprehension

#continuos-monad-transformer:: (a->m r) -> m r  ##simply call -> callback(with-provided-val)
#Mother-of-all-monads: continuation monad (observer pattern)  ~ dual is list-monad (iterator pattern)

def noneMonad(Monad):
	def __init__(self,arg):
		self.arg=arg
	def bind(self, func):
			if arg != None:
				next_arg=func(arg)
				return eventMonad(next_arg)
			else:
				return eventMonad(None)

#promise -> container


