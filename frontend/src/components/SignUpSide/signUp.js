import axios from 'axios';

const signUp2 = async (email, password) => {
    const response = await axios.post('http://127.0.0.1:8000/signup', 
        {
            'username': email,
            'password': password
        },
        {
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    );
    console.log("response status code: " + response.status);
    console.log(response.error)
    if (response.error) console.log("Error: el email ingresado no se encuentra disponible");
    else console.log("Registración exitosa.");
}

const signUp = (email, password) => {
    axios.post('http://127.0.0.1:8000/signup', 
        {
            'username': email,
            'password': password
        },
        {
            'headers': {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            console.log("Response: " + JSON.stringify(response));
            
        })
        .catch(error => {
            console.log("Error: " + JSON.stringify(error));
        });

    // console.log("response status code: " + response.status);
    // console.log(response.error)
    // if (response.error) console.log("Error: el email ingresado no se encuentra disponible");
    // else console.log("Registración exitosa.");
}


export default signUp;