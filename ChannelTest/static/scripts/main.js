var isclassic = true;
var isevaluated = false;
var redundancy = 1;
var error_prob = 0;
var single = true;
var entanglement=8;

function updateEntanglement(value){
    entanglement = value;
    var enttitle = document.getElementById("EntanglementTitle");
    enttitle.innerHTML = `Entaglement Coefficient: ${1/value}`;
    EvaluateImage();
}



function handleClassical(){
    if (!isclassic){
        isevaluated = false;
    }
    isclassic = true;

    var toggle = document.getElementById("QuantumButton");
    toggle.hidden = true;
    var title = document.getElementById("TitleName");
    title.innerHTML = "Classical Channel";
    var error_dis = document.getElementById("ErrorCont");
    error_dis.hidden = false;
    EvaluateImage();
};

function handleQuantum(){
    if (isclassic){
        isevaluated=false;
    }
    var toggle = document.getElementById("QuantumButton");
    toggle.hidden = false;
    isclassic = false;
    var title = document.getElementById("TitleName");
    title.innerHTML = "Quantum Channel";
    var error_dis = document.getElementById("ErrorCont");
    error_dis.hidden = true;
    EvaluateImage();
}


function ToggleChannels(){
    single = !single;
    isevaluated = false;
    var qbutt = document.getElementById('Qbutt');
    qbutt.innerHTL = single? "Toggle Product Channel" : "Toggle Single Channel";
    EvaluateImage();
}

function compute_quantum(){
    let image = document.getElementById("OutputImage");
    if (single){
        image.src = "/images/static.png";
    }else{
        console.log("The thing here");
        image.src = `/images/quantum-${entanglement}-${redundancy}.png`;
    }
    console.log("Handle quantum");
}


function compute_classic(redundancy, error_prob){
    let image = document.getElementById("OutputImage");
    image.src = `/images/classical-${error_prob*100}-${redundancy}.png`;
}


function EvaluateImage(){
    if (!isevaluated){
        if (isclassic){
            compute_classic(redundancy, error_prob);
        } else{
            compute_quantum(redundancy, single);
        }
    }
}

function updateError(value){
    if (error_prob != value/10){
        isevaluated = false;
    }
    error_prob = value/10;
    var errtitle =document.getElementById("ErrorTitle");
    errtitle.innerHTML = `Error Probability: ${error_prob}`;
    EvaluateImage();
}


function updateRedundancy(value){
    if (2*value + 1 != redundancy) {
        isevaluated = false;
    }
     redundancy = 2*value + 1;
     var redtitle = document.getElementById("RedundancyTitle");
    redtitle.innerHTML = `Redundancy: ${redundancy}`;
    EvaluateImage();
};







