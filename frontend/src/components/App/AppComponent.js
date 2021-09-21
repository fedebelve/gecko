import React from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import Basic from '../ImageLoadScreen/DropZone/Basic';
import PageNotFound from '../PageNotFound/PageNotFound';
import SignInSide from '../SignInSide/SignInSide';
import SignUpSide from '../SignUpSide/SignUpSide';

const AppComponent = () => {
    return (
        <div className="App" id="root">
            <Switch>
                <Route exact path='/' render={() => <SignInSide />} />
                <Route exact path='/login' render={() => <SignInSide />} />
                <Route exact path='/home' render={() => <Basic />} />
                <Route exact path='/signup' render={() => <SignUpSide />} />
                {/* <Route exact path='/signup' render={() => <Sign handleLogin={handleLogin} />} /> */}
                <Route component={PageNotFound} />
            </Switch>
        </div>
    );
}

export default AppComponent;