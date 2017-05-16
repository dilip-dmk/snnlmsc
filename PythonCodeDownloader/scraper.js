/**
 * Created by madhukum on 17.02.17.
 */
var Trending = require("github-trend");
var scraper = new Trending.Scraper();
var fs = require('fs');
var util = require('util');
var log_file = fs.createWriteStream(__dirname + '/daily_repo.txt', {flags: 'a'});
var log_stdout = process.stdout;
var cp = require('child_process');

console.log = function (d) { //
    log_file.write(util.format(d) + '\n');
    log_stdout.write(util.format(d) + '\n');
};

scraper.scrapeTrendingRepos("python").then(function (repos) {
    repos.forEach(function (repo) {
        console.log('https://github.com/' + repo.owner + '/' + repo.name);
    });
}).catch(function (err) {
    console.log(err.message);
});

cp.fork(__dirname + '/dup_repos.js');