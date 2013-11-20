
//data
var pr = [20,40,60,80,100]
var emp=[0]
var BTNDATA = {'B0':[0],'B1':[0],'B2':[0],'B3':[0],'B4':[0]}
var ky = Object.keys(BTNDATA);



//dom

for(i=0;i<5;i++) {
	domBtn( pr[i], ky[i] )
}

d3.select('body').append('br');

var margin = {top: 20, right: 20, bottom: 30, left: 40},
	    width = 960 - margin.left - margin.right,
		    height = 500 - margin.top - margin.bottom;

var svg1 = d3.select('body').append('svg')
			 .attr("width", width + margin.left + margin.right)
    		.attr("height", height + margin.top + margin.bottom)
	      	.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

function domBtn(pr,b){
	var dbn =  dom("BUTTON", {id: ""+b, class:"btn"},pr+"%");
	document.body.appendChild( dbn )
}
var T=0;
function main(){	
	var i, crldt;
	for( i = 0; i < 5; i++ ) {
			//domBtn( pr[i], ky[i] ) 
			addBtnClk( ky[i] )
	}
}
//}

main();


function sum2(dct){
	var T=0,
		i;

	for(i=0;i<5;i++) {
		T += dct[i];
	}
		
	//console.log("test --sum ", "total ", T)
	return T;
}

var TOT = [0,0,0,0,0]
function addBtnClk(bid) {
	var i; 
	i = bid.slice(-1)
	d3.select('#'+bid).on("click", function() {
		if( T >= 15 )
			alert("max reached")
		else
			BTNDATA[bid].push(1);
			TOT[i] += 1
			T = sum2(TOT)
			console.log("### data total ", T );
			var crldt = BTNDATA[bid].slice()
			addCrl( bid, crldt );
	});
}
function addCrl( bi, crldt) {
	var i, btni, crl, shm, drow;

	//data
	btni = bi.slice(-1)
	drow = BTNDATA[bi].slice()
	shm = 0;	
	shm = sum( crldt );
	//T += shm;

	crl = svg1.selectAll(".circle")
				.data(drow)
	//update

	//enter
	crl.enter()
		.append("circle")
		.attr('class', function(){ return 'c'+btni})
		.attr('cx', function() {
			return btni * 50 
		})
		.attr('cy', function(){
			console.log( " --cy data ", shm, T )
			return height - (drow.length * 30) ;
		} )
		.attr("r",13)
		.style("fill","red")
		
	//exit	
	dlt(crl,bi)
	crl.exit()
		.remove()

}



function dlt(crl,bi){
	crl.on("click", function(){
			BTNDATA[bi].pop();
		});
}
	
	

function sum( dct ){
	var T=0,i; //dct={};
	//var dct = dslice.slice();
	//console.log("## len2 ",dct.length)
	for(i=0; i < dct.length; i++) {
		//console.log("len ",dct.length)
		T += dct[i];
		}
	return T;
}

dd ={"a":[0], "b":[0,1]}

//for (i in dd)
//	console.log("$$$ ", test, dd.i, dd[i])

var z = 'a'
dd[z].push(1)
dd[0] = 1
console.log("-- dd ", dd[0])
dd.b.push(1)
dd.b.push(1)
dd.b.push(1)
nd = dd.a.slice()


var sdd = sum( nd )
//console.log("!!--test sum", sdd )




function addGroups(){
	var groups={};
	for(var i=0;i<5;i++){
		var bid = 'B'+i;
	  	groups[bid]= svg1.append("g")
	}
	return groups;
}
