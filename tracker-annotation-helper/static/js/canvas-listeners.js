// Handle mouse events
canvas.addEventListener('mousedown', function (event) {
    let shouldCreateNewSquare = true;

    squares[frame].forEach((square, i) => {
        // Check if the mouse is over the resize handle
        if (square.isOverResizeHandle(mouseX, mouseY)) {
            square.setAsResizing(mouseX, mouseY);
            shouldCreateNewSquare = false;
        }
        // Check if the mouse is over the square
        else if (square.isMouseOver(mouseX, mouseY)) {
            square.isDragging = true;
            lastX = mouseX;
            lastY = mouseY;
            shouldCreateNewSquare = false;
        }
    });

    if (shouldCreateNewSquare) {
        let square = new Square(mouseX, mouseY,
            Square.getHandleSize(), Square.getHandleSize());
        squares[frame].push(square);
        square.setAsResizing(mouseX, mouseY);
    }
});


canvas.addEventListener('mousemove', function (event) {
    let shouldRedraw = false
    squares[frame].forEach(square => {
        shouldRedraw = shouldRedraw || square.update(mouseX, mouseY);
    });

    //if (shouldRedraw) drawSquares();
    drawSquares();
});


canvas.addEventListener('mouseup', function (event) {
    squares[frame].forEach(square => {
        if (square.isResizing && square.number == "-1") {
            let ids = squares[frame].map(square => square.number);
            let number;
            while (number = prompt("ID number: ")) {
                if (isNaN(number)) {
                    alert("Not a number");
                } else if (ids.includes(number)) {
                    alert("Number already in ids");
                } else {
                    break;
                }
            }
            if (number == null) {
                alert("Operation Aborted");
            } else {
                square.setNumber(number);
            }
            drawSquares();
        }

        square.isDragging = false;
        square.isResizing = false;
        square.resizeHandle = null;
    })
});


canvas.addEventListener('mousemove', (event) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = event.clientX - canvas.offsetLeft + window.scrollX;
    mouseY = event.clientY - canvas.offsetTop + window.scrollY;
});


canvas.addEventListener('keydown', function (event) {
    if (event.key === 'd') {
        squares[frame].forEach((square, index) => {
            if (square.isMouseOver(mouseX, mouseY)) {
                squares[frame].splice(index, 1);
                drawSquares();
            }
        });
    }
});