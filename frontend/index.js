const url = 'http://127.0.0.1:5000/predict-fraud?'

const trans_data_card = document.getElementsByClassName('colums')[0]
const labels = ['Time', 'Amount', 'use_chip', 'MCC', 'transaction_error', 'age',
                  'ret_age', 'Gender', 'Zipcode', 'yearly_income', 'total_debt', 'fico_score']
const submitbtn = document.getElementById('submit')
submitbtn.onclick = getData
labels.forEach(addLabels)

async function getFraudProbabilty(data) {
  params = new URLSearchParams(data)
  const response = await fetch(url + params,{
    method: 'GET',
    headers: {'Content-type': 'application/json; charset=UTF-8'}})
  const json = await response.json()
  const prob_tag = document.getElementById('prb-tag')
  let prob = Number(json)*100
  if (prob < 0.001) {
    prob_tag.innerText = prob_tag.innerText +' Negligible'
  }
  else {
  prob_tag.innerText = prob_tag.innerText + String(prob) + '%'}
}

function addLabels(label){
  let div = document.createElement('div')
  if (label == 'transaction_error') {
    div.innerHTML = `<label for="${label}"> Transaction Error<span>*</span></label>
          <select name="transaction-error" id="${label}">
          <option value="None">None</option>
          <option value="Insufficient Balance">Insufficient Balance</option>
          <option value="Bad PIN">Bad PIN</option>
          <option value="Bad CVV">Bad CVV</option>
          <option value="Bad Expiration Date">Bad Expiration Date</option>
          <option value="Bad Card Number">Bad Card Number</option>
          <option value="Technical Glitch">Technical Glitch</option></select>`} 
  else if (label == 'use_chip'){
    div.innerHTML = `<label for="${label}"> Transaction Type<span>*</span></label>
          <select name="transaction-error" id="${label}">
          <option value="Swipe Transaction">Swipe Transaction</option>
          <option value="Chip Transaction">Chip Transaction</option>
          <option value="Online Transaction">Online Transaction</option></select>`}
  else if (label == 'Gender'){
    div.innerHTML = `<label for="${label}">User Gender<span>*</span></label>
          <select name="transaction-error" id="${label}">
          <option value="Male">Male</option>
          <option value="Female">Female</option></select>`
  }
  else {
    const label_names = {Time:'Transaction Time',Amount:'Transaction Amount',
          age:'User Age', ret_age:'User Retairment Age', Zipcode:'User Zipcode',
          yearly_income:'User Yearly Income', total_debt:'User Total Debt', 
          fico_score:'User FICO score', MCC:'User MCC'}
    div.innerHTML = `<label for="${label}"> ${label_names[label]}<span>*</span></label>
          <input id="${label}" type="text" name="${label}" required/>`}
  div.setAttribute('class', 'item')
  trans_data_card.appendChild(div)
}

function getData() {
  document.getElementById('prb-tag').innerText = 'Fraud Probability: '
  let data = {}
  const items = document.getElementsByClassName('item')
  labels.forEach(function(label) {
    const item = document.getElementById(label)
    data[item.id] = item.value
  })
  getFraudProbabilty(data)
}

