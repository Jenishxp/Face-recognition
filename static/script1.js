import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword, GoogleAuthProvider, signInWithPopup } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-auth.js"; 

// Your firebaseConfig
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

// New: Google Provider
const provider = new GoogleAuthProvider();

// Handle Google login
//const googleLoginBtn = document.getElementById('google-login');

// googleLoginBtn.addEventListener('click', function() {
//     signInWithPopup(auth, provider)
//     .then((result) => {
//         const user = result.user;
//         alert("Google sign-in successful! Welcome, " + user.displayName);
//         console.log(user); // Optional: log user details
//         window.location.href = "index.html"; // redirect if you want
//     })
//     .catch((error) => {
//         console.error(error.code, error.message);
//         alert("Google sign-in failed: " + error.message);
//     });
// });


const googleLoginBtn = document.getElementById('google-login');

googleLoginBtn.addEventListener('click', function() {
    signInWithPopup(auth, provider)
    .then((result) => {
        const user = result.user;
        alert("Google sign-in successful! Welcome, " + user.displayName);
        console.log(user);
        window.location.href = "/dashboard"; // redirect if you want
    })
    .catch((error) => {
        console.error(error.code, error.message);
        alert("Google sign-in failed: " + error.message);
    });
});



// createUserWithEmailAndPassword.addEventListener('click', function() {
//     signInWithPopup(auth, provider)
//     .then((result) => {
//         const user = result.user;
//         alert("user created " + user.displayName);
//         console.log(user); // Optional: log user details
//         window.location.href = "dashboard.html"; // redirect if you want
//     })
//     .catch((error) => {
//         console.error(error.code, error.message);
//         alert("failed: " + error.message);
//     });
// });

const submitBtn = document.getElementById('submit'); // your "Sign Up" button

submitBtn.addEventListener('click', function(event) {
    event.preventDefault(); // stop form submitting

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
        const user = userCredential.user;
        alert("User created: " + user.email);
        console.log(user);
        window.location.href = "/dashboard"; // redirect after signup
    })
    .catch((error) => {
        console.error(error.code, error.message);
        alert("Failed to create account: " + error.message);
    });
});
