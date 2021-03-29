<?php  
$conn = new mysqli('localhost', 'root', '');  
mysqli_select_db($conn, 'counting');  
$sql = "SELECT `motor`,`mobil`, `bus_or_truk`, `waktu`  FROM `result`";  
$setRec = mysqli_query($conn, $sql);  
$columnHeader = '';  
$columnHeader = "Motor" . "\t" . "Mobil" . "\t" . "Bus or Truck" . "\t" . "Waktu" . "\t" ;  
$setData = '';  
  while ($rec = mysqli_fetch_row($setRec)) {  
    $rowData = '';  
    foreach ($rec as $value) {  
        $value = '"' . $value . '"' . "\t";  
        $rowData .= $value;  
    }  
    $setData .= trim($rowData) . "\n";  
}  
  
header("Content-type: application/octet-stream");  
header("Content-Disposition: attachment; filename=Laporan.xls");  
header("Pragma: no-cache");  
header("Expires: 0");  

  echo ucwords($columnHeader) . "\n" . $setData . "\n";  
 ?> 
 