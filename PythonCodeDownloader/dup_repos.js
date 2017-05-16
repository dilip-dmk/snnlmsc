/**
 * Created by madhukum on 17.02.17.
 */
var neek = require('neek');

var readable = './daily_repo.txt';
var writable = './repos.txt';

neek.unique(readable, writable, function (result) {
    console.log(result);
});
