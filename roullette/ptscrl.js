//patient scroll data
// javascript:


var	svg2 = d3.select('body').append('svg').attr('width',500).attr('height',400); //.attr("id","d3svgctnr");
var margin = {top: 10, right: 50, bottom:20, left:50},
	height = 500 - margin.top - margin.bottom;
	width = 120 - margin.left - margin.right;

var min = Infinity,
	max = -Infinity;

//data
//{sys,dia,ox,hr1,hr2,wt}
//var sys=[],dia=[],data=[];
//d3.json("data0.json", function(data){
//	data.map(function(d) {
//		var ky = d3.keys(d)[0];
//		d[ky].ky = ky;
//
//		sys.push(d.SYS);
//		dia.push(d.DIA);
//		data.push(d.TIME);
//
//		return d[ky];
//
//	});
//});

//dom
var w = 15,
	h = 70;
var x =d3.scale.linear()
		.domain([0,1])
		.range([0,w]);
var y = d3.scale.linear()
			.domain([0,1])
			.rangeRound([0,h]);

//rectangle
//move up
function rndrMvBx(fl) {
	//data
	d3.json(fl, function(dt){
		var data = []
		var i=0
		
		dt.map(function(d) {
			//dt.slice(i,-w)
			//console.log(d);
			for(i; i<w; i++){
				console.log(d);
				data.push(d);
			}
			console.log( "LEN: " , data.length )
		});
		//join new data to old nodes	
		var rect= svg2.selectAll('rect')
			.data(data, function(d) { return d.TIME; })
		//enter
			rect.enter().insert('rect','line')
			.attr("class", "box")
			.attr("x", function(d,i) { return x(i) - .5; })
			.attr("y", function(d) { return h - y(d.SYS - d.DIA ) - .5;} )
			.attr("width", w) //idth + margin.left + margin.right)
			.attr("height", function(d) { return y(d.SYS - d.DIA); })
			//.attr("translate", function(d) { return y
		//update
			rect.transition()
			.duration(1000)
			.attr("x", function(d,i) { return x(i) - 0.5 })
		//exit
			rect.exit().remove();

});
}
rndrMvBx("data0.json");

//setInterval(function() {
//	rndrMvBx("data0.json");
//	data.shift();
//	//data.push;
//}, 1500);
	





//trending data

/*
//function main(){
// insert data
d3.csv("telehealth.csv", function(error, csv) {
  var data = [];

  csv.forEach(function(x) {
    var e = Math.floor(x.Expt - 1),
        r = Math.floor(x.Run - 1),
        s = Math.floor(x.Speed),
        d = data[e];
    if (!d) d = data[e] = [s];
    else d.push(s);
    if (s > max) max = s;
    if (s < min) min = s;
  });


});

	// svg
	drawBox();	
//}
//main();

//dynamic dom elements
var pBox = d3.box()
	.whiskers(iqr(1.5))
	.width( width )
	.height( height );
var w = 20,
	h = 80;
var x =d3.scale.linear()
		.domain([0,1])
		.range([0,w]);
var y = d3.scale.linear()
			.domain([0,1])
			.rangeRound([0,h]);

function drawBox(){
		//timestamp
		svg2.selectAll('pBox')
			.data(data, function(d) { return d.time; })
		//enter
		pBox.enter().insert('svg')
			.attr("class", "box")
			.attr("x", function(d,i) { return x(i) - .5; })
			.attr("y", function(d) { return h - y(d.value) - .5;} )
			.attr("width", width + margin.left + margin.right)
			.attr("height", function(d) { return y(d.value); })
		//update
		pBox.transition()
			.duration(1000)
			.attr("x", function(d,i) { return x(i) - 0.5 });
		//exit
		pBox.exit().remove();
	}

// Returns a function to compute the interquartile range.
function iqr(k) {
  return function(d, i) {
    var q1 = d.quartiles[0],
        q3 = d.quartiles[2],
        iqr = (q3 - q1) * k,
        i = -1,
        j = d.length;
    while (d[++i] < q1 - iqr);
    while (d[--j] > q3 + iqr);
    return [i, j];
  };
}

//function axisScrl()
//
//
////function clickNext()
//
//
///*
//# python:
//function getHard(): pass
//		
//
//
//function writeD(): pass
//
//
//
//function boxTopBlwBp(dd): pass
//
//
//
//*/
