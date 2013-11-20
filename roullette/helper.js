//helper functions from Haverbeke's Eloquent JS
//
//


function forEach(array, action) {
	for ( var i =0; i<array.length; i++)
		action(array[i]);
}

//test forEach(["wamp","foma","gran"]), print);

function tprint(fnc){
	console.log("test --", fnc)
}
function dprint(fnc){
	console.log("dbg --", fnc)
}

function partial(func) {
	var knownArgs = arguments;
	return function() {
		var realArgs = [];
		for (var i =1; i<knownArgs.length; i++)
			realArgs.push(arguments[i]);
		return func.apply(null, realArgs);
	};
}

//test map(partial(op["+"], 1), [0,2,4,6,8,10]);

function Dictionary(startValues) {
	this.values = startValues || {};
}

Dictionary.prototype.store = function(name,value) {
	this.values[name] = value;
};
Dictionary.prototype.lookup = function(name) {
	return this.values[name];
};
Dictionary.prototype.contains = function(name) {
	return Object.prototype.propertyIsEnumerable.call(this.values,name);
};
Dictionary.prototype.each = function(action) {
	forEachIn(this.values, action);
};

function bind(func,object) {
	return function(){
		return func.apply(object,arguments);
	};
}

function method(object,name){
	return function(){
		object[name].apply(object,arguments);
	};
}

Object.prototype = function(baseConstructor) {
	this.prototype = clone(baseConstructor.prototype);
	this.prototype.constructor = this;
};
Object.prototype.method = function(name,func) {
	this.prototype[name] = func;
};

//test function Strange(){}
//Strange.inherit(Array);
//Strange.method("push",function(value){
//	Array.prototype.push.call(this,value);
//	Array.prototype.push.call(this,value);
//});
//var strange = new Strange();
//strange.push(4);

Object.prototype.create = function() {
	var object = clone(this);
	if (object.construct != undefined)
		object.construct.apply(object,arguments);
	return object;
};

Object.prototype.extend = function() {
	var result = clone(this);
	forEachIn(properties,function(name,value) {
		result[name] = value;
	});
	return result;
};



//web
function isTextNode(node) {
	return node.nodeType == 3;
}
//test isTextNode(document.body.firstChild.firstChild.nodeName)
//isTextNode(document.body.firstChild.firstChild.nodeValue)

function dom(name,attributes){
	var node = document.createElement(name);
	if(attributes) {
		forEachIn(attributes, function(name,value){	
			node.setAttribute(name,value);
		});
	}
	for (var i =2; i<arguments.length; i++) {
		var child=arguments[i];
		if (typeof child == "string")
			child = document.createTextNode(child);
		node.appendChild(child);
	}
	return node;
}

function removeNode(node) {
	node.parentNode.removeChild(node);
}
function insertBefore(newNode, node) {
	node.parentNode.insertBefore(newNode,node);
}
//test
//var output = dom("DIV",{id:"printOut"}, dom("H1",null,"print out:"));
//document.body.appendChild(output);
//function print() {
//	var result = [];
//	forEach(arguments, function(arg){result.push(String(arg));});
//	output.appendChild(dom("PRE", null, result.join("")));
//	}

function registerEventHandler(node,event,handler) {
	if(typeof node.addEventListener == "function")
		node.addEventListener(event, handler, false);
	else
		node.attachEvent("on" + event, handler); //internet explorer
}

function unregisterEventHandler(node,event,handler){
	if(typeof node.removeEventListener == "function")
		node.removeEventListener(event,handler,false);
	else
		node.detachEvent("on" + event, handler);
}


function normalizeEvent(event) {
	if (!event.stopPropagation) {
		event.stopProgpagation = function() { this.cancelBubble = true;};
		event.preventDefault = function() { this.returnValue = false;};
	}
	if (!event.stop)
		event.stop = function() {
			this.stopPropagation();
			this.preventDefault();
		};

	if (event.srcElement && !event.target)
		event.target = event.srcElement;
	if ((event.toElement || event.fromElement) && !event.relatedTarget)
		event.relatedTarget = event.toElement || event.fromElement;
	if (event.clientX != undefined && event.pageX  == undefined) {
		event.pageX = event.clientX + document.body.scrollLeft;
		event.pageY = event.clientY + document.body.scrollTop;
	}
	if (event.type == "keypress")
		event.character = String.fromCharCode(event.charCode || event.keyCode);
	return event;
}

function addHandler(node,type,handler){
	function wrapHandler(event) {
		handler(normalizeEvent(event || window.event));
	}
	registerEventHandler(node,type,wrapHandler);
}

function removeHandler(object){
	unregisterEventHandler(object.node,object.type,object.handler);
}

function forEachIn(object,action){
	for (var property in object) {
		if (Object.prototype.hasOwnProperty.call(object,property))
			action(property,object[property]);
	}
}
