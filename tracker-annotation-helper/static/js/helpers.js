function loadData() {
    fetch('/load_data')
        .then(res => res.json()
            .then(data => {
                for (frame in data) {
                    squares[frame] = data[frame].map(s => {
                        return new Square(s.bb_left, s.bb_top,
                            s.bb_width, s.bb_height,
                            s.id)
                    });
                }
            })
            .catch(err => console.log(err)))
        .catch(err => console.log(err));
}


function getFrameData() {
    fetch('/get_current_frame')
        .then(res => res.json()
            .then(json => {
                frame = json.frame;
                document.getElementById("curr-frame").innerHTML = frame;
                if (!(frame in squares)) {
                    console.log(frame);
                    squares[frame] = [];
                }
                drawSquares();
            })
            .catch(err => console.log(err)))
        .catch(err => console.log(err));
}


function loadImage(nextOrPrev) {
    if (nextOrPrev == "prev") {
        return loadSelectedImage(frame - 1);
    } else if (nextOrPrev == "next") {
        return loadSelectedImage(frame + 1);
    }

    return;
}


function loadSelectedImage(frame) {
    fetch(`/get_image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frame })
    })
        .then(
            res => res.blob()
                .then(imageBlob => {
                    const imageUrl = URL.createObjectURL(imageBlob);
                    img.src = imageUrl;
                    img.onload = getFrameData;
                })
                .catch(err => console.log(err)))
        .catch(err => console.log(err))
}


function exportToMOT(filename) {
    let result = "";

    Object.keys(squares).forEach(frame => squares[frame].forEach(square => {
        const id = square.number;
        const x = square.x;
        const y = square.y;
        const w = square.width;
        const h = square.height;

        result += `${frame}, ${id}, ${x}, ${y}, ${w}, ${h}, -1, -1, -1, -1\n`;
    }));

    fetch('/save_square', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            squares: result
        })
    })
        .then(response => response.json()
            .then(data => {
                console.log(data);
                alert(data.msg)
            })
            .catch(error => console.error(error)))
        .catch(error => console.error(error));
}


function drawSquares() {
    if (squares[frame].length != 0) {
        squares[frame].forEach((square, i) => square.draw(canvas, ctx, img, i == 0));
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
    }
}


function promptInt(message) {
    let number;
    while (number = prompt(message)) {
        if (isNaN(number)) {
            alert("Not a number");
        } else {
            break;
        }
    }
    return parseInt(number);
}


function median(values) {
    if (values.length === 0) return 0;

    values.sort(function (a, b) { return a - b; });

    var half = Math.floor(values.length / 2);

    if (values.length % 2)
        return values[half];

    return (values[half - 1] + values[half]) / 2.0;
}


function linearInterpolation1D(x1, x2, nPoints) {
    let step = (x2 - x1) / (nPoints + 1);
    return [...Array(nPoints).keys()].map(n => x1 + step * (n + 1));
}


function linearInterpolation(id) {
    let matchedSquares = [];
    let frames = [];
    for (const f in squares) {
        for (const square of squares[f]) {
            if (square.number != id)
                continue;

            matchedSquares.push(square);
            frames.push(parseInt(f));
            break;
        }
    }

    let newFrames = []
    let newSquares = []
    for (let i = 0; i < frames.length - 1; i++) {
        newSquares.push(matchedSquares[i]);
        newFrames.push(frames[i]);

        let currFrame = frames[i]
        let nextFrameN = frames[i + 1];
        if (currFrame + 1 == nextFrameN)
            continue;

        let xs = matchedSquares.map(s => s.x);
        let ys = matchedSquares.map(s => s.y);
        let ix = linearInterpolation1D(xs[i], xs[i + 1], nextFrameN - currFrame);
        let iy = linearInterpolation1D(ys[i], ys[i + 1], nextFrameN - currFrame);

        let medianW = median(matchedSquares.map(s => s.width));
        let medianH = median(matchedSquares.map(s => s.height));
        for (let j = 0; j < ix.length; j++) {
            let s = new Square(ix[j], iy[j], medianW, medianH, id);
            newFrames.push((j + 1) + currFrame);
            newSquares.push(s);
        }
    }

    for (let i = 0; i < newFrames.length - 1; i++) {
        let f = newFrames[i];
        let found = false;
        for (const square of squares[f]) {
            if (square.number != id)
                continue;
            
            found = true;
            break;
        }
        if (found) continue;
        squares[f].push(newSquares[i]);
    }
}
