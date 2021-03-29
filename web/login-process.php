<?php
session_start();
// db config -------------------------------------------------
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "counting";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 
//----------------------------------------------------------
if ( isset($_POST['login'])){
    $username = $_POST['username'];
	$password = md5( $_POST['password']);

    $sql_query = "SELECT name, email, username, id FROM users 
                  WHERE username='$username' AND password='$password'
                  LIMIT 1"; //sql query
   
	$result = $conn->query($sql_query);
	
	if ($result->num_rows > 0) {
    // data valid
    while($row = $result->fetch_assoc()) {
        	$_SESSION['name'] = $row["name"];
			
			$conn->close();
			header('location:main_pages/index.php');//dirrect to main page
			exit();
		}
	} else { //data invalid
		//alert then back to login page
		echo "<script language=\"javascript\">alert(\"Invalid username or password\");
		document.location.href='index.php?error_login';</script>";
		exit();
	}
  
} else {
    header('location:index.php');
    exit();
}
?>