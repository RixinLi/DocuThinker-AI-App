import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
}

// const firebaseConfig = {
//   apiKey: "AIzaSyDiUOK7QNZgFe6v6PjIYKR7WM6hWiD36l0",
//   authDomain: "docuthinker-ai-app.firebaseapp.com",
//   projectId: "docuthinker-ai-app",
//   storageBucket: "docuthinker-ai-app.firebasestorage.app",
//   messagingSenderId: "40097729609",
//   appId: "1:40097729609:web:fe82dc12b92f18816e8398",
//   measurementId: "G-0XL030C3KD"
// };

// initialization
const app = initializeApp(firebaseConfig);

// export instance
export const auth = getAuth(app);