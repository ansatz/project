/* roulette method
 * gui - 
 * 		d3 vitals graph 
 * 		roulette graph
 * 		Buttons array
 *
 *
 */

var N = 10;
var B = 5;
var BUTTONS = {}

function bb(){
	//instantiate the BUTTONS dictionary
	var i;

	for(i=0; i<B; i++){
		BUTTONS[i]=0
	}
	console.log('BUTTONS ', BUTTONS)
}

function expandBTNS(){
	var i,j, exp={}, vlu;
	for ( i in BUTTONS )
		for( j=0; j < BUTTONS[i] ; j++ )
			vlu = [ i, j*2 ]
			exp[i+''+j] = vlu;
	return exp
}
function dta() {
	var dataset = {}, keyz, vlu, i, j ;
	function z(){
		for( i=0; i<B; i++ )
			console.log("!! dta buttons[i] ", BUTTONS[i])
			for(j=0 ; j<= BUTTONS[i] ; j++)
				keyz = i+""+j;
				vlu = [  i,  j * 2  ];
				console.log( " !!dbg --dta() BTN.len, keyz, vlu", BUTTONS.length , keyz, vlu )
				return dataset[keyz] =  vlu
	};
	console.log('dta() ')
	return z;
}

function sum(dct){
	var T=0,
		i;

	for(i=0;i<B;i++) {
		T += dct[i];
	}
		
	//console.log("test --sum ", "total ", T)
	return T;
}
//event-handler
function lpBtn(i) {
	var btn = document.getElementById("b"+i);
	
	return addHandler(btn,"click", function(event){
				if( T >= N )
					alert("max-limit reached");
				else
					BUTTONS[i] += 1;
					T = sum(BUTTONS);
					//addSVG("b"+i);
					//addSVG2("b"+i);
					console.log("test --lpBtn(i) total T = ",T, BUTTONS[i] );
					btnd3(dset);
					//dta();
					//console.log('dbg --btnd3 :dataset ', dta);	
			});
}

d3.select('#b1')
	.on("click", function(d){
		d3Btn(1);	
	})


function d3Btn(i) {
	var i,btn, inx, T, dtaBtn, ptid;
	
	btn = d3.select("#b"+i)
	//console.log("dbg btn-id", "b"+i, btn);
	btn.on('click', function( d ) {

		console.log("dbg btn-id d ", d);
		if(T >= N)
			alert("max-limit-reached-d3");
		else
			BUTTONS[i] += 1;
			T = sum(BUTTONS)

			//dtaBtn = dta()  //get data into dict{key=ij, val=[i][j*2]
			dtaBtn = expandBTNS();
			
			console.log('## BTN 0', BUTTONS[0], dtaBtn[0] );
			
			//svg// entries return array of dicts[ {key:key,value:value},...{kn,vn} ]
			svg.selectAll("circle")
				.data(d3.entries( dtaBtn  ))  //update
				.enter()					  //enter	
				.append("circle")
				.attr('id', function(d){ 
					console.log('## data id ', d.key)
					console.log("!! dbg d3 --id d[key] ",d.key);
					return d.key; 
				}) 
				.attr('cx', function(d){ 
					console.log("!! dbg d3 --cx ",d.value[0]);
					return d.value[0]; 
				} )
				.attr('cy', function(d){
					console.log("!! dbg d3 --cy",d.value[1]);
					return d.value[1];
				})
				.attr("r",5)
				.style("color","red")
				.on("click", function(){
					ptid = d3.select(this).attr("id");
					delete dtaBtn[ptid];
					//dtaBtn()
				
				})

			//svg.exit().remove()
			//console.log("!!test --btnd3 ", dtaBtn["03"][0],dtaBtn["03"][1])
		}
	);
}

//dom
function addSVG(bi){
	var i = bi.slice(-1),
		stk;

	console.log('dbg --addSVG ', i )	
	dom("SVG", {id: "red-crl" + N}) 

	stk = document.getElementById('red-crl'+N)
	stk.style.left=(i*100)+"px"
	stk.style.top=(100-2*N)+"px"
	
	console.log('test --addSVG, button ',i, ' pressed', stk)
}

//function addSVG2(btnPress){
//	 var NS = "http://www.w3.org/2000/svg",
//		 svg = document.createElementNS(NS,"circle"),
//		 colIndx = btnPress.slice(-1),
//		 posL,posT; //position from left, top
//	 		
//	 posL = colIndx*120;
//	 posT = 100 - (15*BUTTONS[colIndx]) //number elements in the column 
//
//	 svg.id="red-crl"+T;
//	 svg.setAttribute("cx", posL);
//	 svg.setAttribute("cy", posT);
//	 svg.setAttribute("r", "15");
//	 svg.setAttribute("fill", "#336699");
//	 
//	 svgContainer.appendChild(svg);
//	 console.log("dbg --addSVG2 container", svgContainer)
//	 //eleSVG = document.getElementByID(svg.id);
//	 //eleSVG.style.left=(i*100)+"px";
//	 //eleSVG.style.top=(100-2*N)+"px";
//	 
//	 //svg.width=w;
//	 //svg.height=h;
//	 
//	 console.log("test --addSVG2", svg.id)
//	 return svg;
//}


//get data into dict{key=ij, val=ij*2
//BUTTONS is dict

var dset = { "01":[10,100], "02":[10,120], "03":[10,140], "11":[150,100], "12":[150,120] }
var	svg = d3.select('body').append('svg').attr('width',500).attr('height',400); //.attr("id","d3svgctnr");
console.log('svg', svg);
function btnd3(dset){
	var	ptid;

	//svg
	svg.selectAll("circle")
		.data(d3.entries(dset))
		.enter()
		.append("circle")
		.attr('id', function(d){ 
			return d[0]; 
		}) 
		.attr('cx', function(d){ 
			return d["value"][0]; 
		} )
		.attr('cy', function(d){
			return d["value"][1];
		})
		.attr("r",5)
		.on("click", function(){
			ptid = d3.select(this).attr("id");
			delete dset[ptid];
		})				
	console.log("!!test --btnd3 ", dset["03"][0],dset["03"][1])
}

function addBTN(pr,j) {
	var bi = "b" + j
	//console.log("dbg --addBTN", pr , j) //bb.style.left = 300 + offset + "px"
	//console.log('test --addBTN(pr):', bi )
	return dom("BUTTON", {id: "b"+j, class:"btn"},pr);
}


//main
function main(){
	var pr = 20,
		i;
	//set up the dom
	//svgContainer = dom("SVG", {id: "svg-container", width:"500px" , height: "400px", top: "100px", left: "100px"}); //,{width:"600px"},{height:"400px"} );
	//console.log("dbg main() svg", svgContainer);
	//document.body.appendChild( svgContainer );
	bb(); //create button dictionary
	
	//main
	T=sum(BUTTONS);
	for(i=0; i<B; i++)
	{
		console.log("dbg --main()", i) ;

		//button
		document.body.appendChild( addBTN(pr+'%',i) )
		pr += 20
	
		//eventhandler
		//lpBtn(i)	//add handlers, svg, etc	
		d3Btn(i);
		console.log("test --main() sum ", sum(BUTTONS), i )

	}	
	//dataset = dta()
	//btnd3(dset);
	console.log("test --main: btnd3()", dset)
}

main()


//d3
function filter(hardjson){
	var i;
	

}

		//pieces = []
		//pieces.push('incrBtn')
		//pieces.push(i)
		//var ib = pieces.join("")

























