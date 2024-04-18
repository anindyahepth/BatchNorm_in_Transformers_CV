(function(){
  const http = new XMLHttpRequest(); 
  var canvas = document.querySelector("#input");
  var ctx = canvas.getContext("2d");
  var mousedown = false; 
  
  const submitButton = document.querySelector(".submit")
  const clearButton = document.querySelector(".clear") 
  
  // get the canvas context 
  ctx.strokestyle = "black"; 
  ctx.lineWidth = 6; 
  ctx.lineJoin= "round"; 
  
  
  function draw_stroke(e){
    if(!mousedown){
      return 
    }
  
    var x = e.clientX - canvas.offsetLeft; 
    var y = e.clientY - canvas.offsetTop; 
  
    ctx.lineTo(x, y); 
    ctx.stroke(); 
    ctx.beginPath(); 
    ctx.moveTo(x,y); 
  
  }
  
  function submitQuery(){

    const image = canvas.toDataURL();
    
    let url = "/predict"
  
    http.open("POST", url) 
  
    http.send(image) 
  
    http.onload = function(){

      let predictionElement = document.querySelector("#class-pred", "#confidence");
      
      
      if (http.status === 200){
        let response_obj = http.response;
        
 predictionElement.textContent = `Recognised Digit with confidence score: ${response_obj} `;
    
        
      }
      
      else{
        predictionElement.textContent = "An error occured";
        
      }
       
    }
  
  }
  
  function clearCanvas(){
    ctx.clearRect(0, 0, canvas.width, canvas.height); 
    let predictionElement = document.querySelector("#class-pred"); 
    predictionElement.textContent = ""; 
  }
  
  // add the event listeners 
  canvas.addEventListener("mousedown", ()=>{mousedown = true}); 
  canvas.addEventListener("mousemove", draw_stroke); 
  canvas.addEventListener("mouseup", ()=>{ mousedown = false; ctx.beginPath()}); 
  submitButton.addEventListener("click", submitQuery); 
  clearButton.addEventListener("click", clearCanvas);
})();
