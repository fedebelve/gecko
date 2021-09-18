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

const SignUpSide = () => {
    const [email, setEmail] = useState('');
    const [passwordA, setPasswordA] = useState('');
    const [passwordB, setPasswordB] = useState('');
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const classes = useStyles();
    const [alertUserNameAlreadyRegistered, setAlertUserNameAlreadyRegistered] = useState(false);
    const [alertPasswordsNotMatch, setAlertPasswordsNotMatch] = useState(false);

    const [redirect, setRedirect] = useState('');
    

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
            setRedirect('/sign-in');
        })
        .catch(error => {
            console.log("Error: " + JSON.stringify(error));
            setAlertUserNameAlreadyRegistered(true);
        });
    }

    const formRedirect = redirect ? <Redirect to={redirect} /> :
        <form className={classes.form} noValidate>
            <TextField onChange={(event) => setFirstName(event.target.value)}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                id="first-name"
                label="First name"
                name="first-name"
                placeholder="Your first name"
                autoFocus
                value={firstName}
            />
            <TextField onChange={(event) => setLastName(event.target.value)}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                id="last-name"
                label="Last name"
                name="last-name"
                placeholder="Your last name"
                value={lastName}
            />
            <TextField onChange={(event) => setEmail(event.target.value)}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                id="email"
                label="E-mail"
                name="email"
                placeholder="email@example.com"
                value={email}
            />
            <TextField onChange={(event) => setPasswordA(event.target.value)}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="password"
                label="Password"
                type="password"
                id="passwordA"
            />
            <TextField onChange={(event) => setPasswordB(event.target.value)}
                variant="outlined"
                margin="normal"
                required
                fullWidth
                name="passwordB"
                label="Confirm password"
                type="password"
                id="passwordB"
            />
            <Button onClick={(event) => {
                    event.preventDefault(); // Para evitar que recargue la página 
                    if (validateFormInput({ passwordA, passwordB }))
                        signUp(email, passwordA);
                }}
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
                    <NavLink to="/sign-in" variant="body2">
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