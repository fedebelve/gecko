import React from 'react';
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import Link from '@material-ui/core/Link';
import { NavLink } from 'react-router-dom';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import { useState } from 'react';
import signUp from './signUp';
import Copyright from '../HomePage/components/Copyright';
import useStyles from './useStyles';
import { Redirect } from 'react-router-dom';
import axios from 'axios';
import { Alert } from '@material-ui/lab';
import { useFormik } from 'formik';

const SignUpSide = () => {
    const [email, setEmail] = useState('');
    const [passwordA, setPasswordA] = useState('');
    const [passwordB, setPasswordB] = useState('');
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [nroDoc, setNroDoc] = useState('');
    const [country, setCountry] = useState('');
    const [birthDate, setBirthDate] = useState('');
    const [jobType, setJobType] = useState('');
    const [institution, setInstitution] = useState('');
    const classes = useStyles();
    const [alertUserNameAlreadyRegistered, setAlertUserNameAlreadyRegistered] = useState(false);
    const [alertPasswordsNotMatch, setAlertPasswordsNotMatch] = useState(false);
    const [redirect, setRedirect] = useState('');
    
    const formik = useFormik({
        initialValues: {
            first_name: '',
            last_name: '',
            email: '',
            password: '',
            doc_number: '',
            country: '',
            birth_date: '',
            job_type: '',
            institution: '',
            passwordA: '',
            passwordB: ''
        },
        onSubmit: values => {
            console.log('Form data to submit', values);
        }
    });

    // console.log('Form values', formik.values);

    const VALID_EMAIL_REGEX = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;
    
    const validatePassword = (passwordA, passwordB) => {
        return passwordA.localeCompare(passwordB) === 0;
    };

    const validateFormInput = ({ passwordA, passwordB }) => {
        if (validatePassword(passwordA, passwordB)) {
            setAlertPasswordsNotMatch(false);
            return true;
        } else {
            setAlertPasswordsNotMatch(true);
            return false;
        }
    };
    
    const signUp = (email, password) => {
        axios.post('http://127.0.0.1:8000/signup', 
        {
            "username": email,
            "password": password,
            "first_name": firstName,
            "last_name": lastName,
            "email": email,
            "nro_doc": nroDoc,
            "country": country,
            "birth_date": birthDate,
            "job_type": jobType,
            "institution": institution
        },
        {
            'headers': {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            console.log("Response: " + JSON.stringify(response));
            setRedirect('/login');
        })
        .catch(error => {
            console.log("Error: " + JSON.stringify(error));
            setAlertUserNameAlreadyRegistered(true);
        });
    }

    const formRedirect = redirect ? <Redirect to={redirect} /> :
        // <form className={classes.form} noValidate>
        <form className={classes.form} noValidate onSubmit={formik.handleSubmit}>
            <TextField onChange={formik.handleChange} value={formik.values.first_name}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                id="first_name"
                label="First name"
                name="first_name"
                placeholder="Your first name"
                autoFocus
            />
            <TextField onChange={formik.handleChange} value={formik.values.last_name}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                id="last_name"
                label="Last name"
                name="last_name"
                placeholder="Your last name"
            />
            <TextField onChange={formik.handleChange} value={formik.values.email}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                id="email"
                label="E-mail"
                name="email"
                placeholder="email@example.com"
            />
            <TextField onChange={formik.handleChange} value={formik.values.passwordA}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="password"
                label="Password"
                type="password"
                id="passwordA"
            />
            <TextField onChange={formik.handleChange} value={formik.values.passwordB}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="passwordB"
                label="Confirm password"
                type="password"
                id="passwordB"
            />
            <TextField onChange={formik.handleChange} value={formik.values.doc_number}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="doc_number"
                label="Numero de documento"
                id="doc_number"
            />
            <TextField onChange={formik.handleChange} value={formik.values.country}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="country"
                label="Pais"
                id="country"
            />
            <TextField onChange={formik.handleChange} value={formik.values.birth_date}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="birthDate"
                label="Fecha de nacimiento"
                id="birthDate"
            />
            <TextField onChange={formik.handleChange} value={formik.values.job_type}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="jobType"
                label="Trabajo"
                id="jobType"
            />
            <TextField onChange={formik.handleChange} value={formik.values.institution}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="institution"
                label="Institución"
                id="institution"
            />
            <Button 
                    // event.preventDefault(); // Para evitar que recargue la página 
                    // if (validateFormInput({ passwordA, passwordB }))
                    //     signUp(email, passwordA);
                // }}
                type="submit"
                fullWidth
                variant="contained"
                color="primary"
                className={classes.submit}
            >
                Sign Up
            </Button>
            {alertUserNameAlreadyRegistered ?
                <Alert severity="error">That email has already been registered. Please enter a different one.</Alert>
                : null
            }
            {alertPasswordsNotMatch ?
                <Alert severity="error">Passwords don't match. Please check those and try again.</Alert>
                : null
            }
            <Grid container>
                <Grid item>
                    <NavLink to="/login" variant="body2">
                        {"Do you already have an account? Sign In"}
                    </NavLink>
                </Grid>
            </Grid>
            <Box mt={5}>
                <Copyright />
            </Box>
        </form>

    return (
        <div className={classes.paper}>
            <Avatar className={classes.avatar}>
                <LockOutlinedIcon />
            </Avatar>
            <Typography component="h1" variant="h5">
                Sign Up
            </Typography>
            {formRedirect}
        </div>
    );
}

export default SignUpSide;