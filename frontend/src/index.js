import React from 'react';
import ReactDOM from 'react-dom';
// import './index.css';
import App from './components/App/App';
import SignIn from './components/SignIn/SignIn';
import reportWebVitals from './reportWebVitals';
import ImageLoadScreen from './components/ImageLoadScreen/ImageLoadScreen';
import ImageUploadButton from "./components/ImageLoadScreen/ImageUploadButton/ImageUploadButton";
import FileList from './components/ImageLoadScreen/DragAndDrop/FileList'
import FileUpload from './components/ImageLoadScreen/FileUpload/FileUpload';
import Basic from './components/ImageLoadScreen/DropZone/Basic';
import Previews from './components/ImageLoadScreen/DropZone/Previews';
import ImageUpload from './components/ImageLoadScreen/ImageUpload/ImageUpload';
import HomePage from './components/HomePage/HomePage';
import SignUpSide from './components/SignUpSide/SignUpSide';
import SignInSide from './components/SignInSide/SignInSide';
import { BrowserRouter } from 'react-router-dom';
import AppComponent from './components/App/AppComponent';
import ImageUpload64 from './components/ImageLoadScreen/ImageUpload/ImageUpload64';

ReactDOM.render(
  <React.StrictMode>
    <BrowserRouter>
      {/* <Basic /> */}
      {/* <Previews/> */}
      {/* <ImageUpload/> */}
      {/* <Basic /> */}
      {/* <HomePage/> */}
      {/* <SignInSide/> */}
      {/* <SignInSide/> */}
      <AppComponent/>
      {/* <ImageUpload64 /> */}
    </BrowserRouter>
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();