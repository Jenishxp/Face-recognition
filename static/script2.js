
// Import the functions you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-app.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-auth.js"; 

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAe8sWNx9oD8A839hrjPWBpEpQ-6Z8aURs",
  authDomain: "login-cf8aa.firebaseapp.com",
  projectId: "login-cf8aa",
  storageBucket: "login-cf8aa.appspot.com",
  messagingSenderId: "209510988116",
  appId: "1:209510988116:web:be451ad5614da76b5470c1"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Handle submit
const submitBtn = document.getElementById('submit');

submitBtn.addEventListener("click", function(event) {
    event.preventDefault();

    const mail = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    signInWithEmailAndPassword(auth, mail, password)
    .then((userCredential) => {
        const user = userCredential.user;
        alert("Account logged in successfully!");
        console.log(user);
        // Optionally redirect to a new page:
        window.location.href = "/dashboard";
    })
    .catch((error) => {
        console.error(error.code, error.message);
        alert("Not Found...Create an account");
    });
});
