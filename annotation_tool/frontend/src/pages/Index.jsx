import React, { useState, useEffect } from "react";
import { Box, Button, VStack, HStack, Text, Heading, Link, Checkbox, Select } from "@chakra-ui/react";
import { Slider, SliderTrack, SliderFilledTrack, SliderThumb, SliderMark } from '@chakra-ui/react'
import "./Index.css";
import { useParams, useNavigate } from 'react-router-dom';


const Index = () => {
  const { id } = useParams();
  let navigate = useNavigate();

  const [url, setUrl] = useState("");
  const [numExamples, setNumExamples] = useState(0);
  const [lines, setLines] = useState([]);
  const [isIgnored, setIsIgnored] = useState(false);
  const [startDragRow, setStartDragRow] = useState(null);
  const [isDragSelecting, setIsDragSelecting] = useState(true);
  const [displayMode, setDisplayMode] = useState("annotations");
  const [threshold, setThreshold] = useState(0.5);

  useEffect(() => {
    fetch(`http://localhost:8000/example/${id}`)
      .then((response) => response.json())
      .then((example) => {
        const preSelectedLines = new Set(example.selected_lines);
        // Slice out last line, since we assume the doc ends with a "\n"
        const linesFromFile = example.content.split("\n").slice(0, -1).map((line, index) => ({
          id: index,
          text: line,
          isSelected: preSelectedLines.has(index),
          predictionScore: example.line_predictions[index]
        }));
        setLines(linesFromFile);
        setNumExamples(example.total_examples);
        setUrl(example.url);
        setIsIgnored(example.is_ignored || false);
      });
  }, [id]);

  const handleMouseDown = (lineIdx) => {
    if (displayMode != "annotations" && displayMode != "loss")
      return;

    document.body.classList.add("no-select");
    const selectionMode = !lines[lineIdx].isSelected;
    setIsDragSelecting(selectionMode);
    setStartDragRow(lineIdx);

    // Select current line
    lines[lineIdx].isSelected = selectionMode;
    setLines([...lines]);
  };

  const handleMouseUp = () => {
    if (displayMode != "annotations" && displayMode != "loss")
      return;

    document.body.classList.remove("no-select");
    setStartDragRow(null);
    saveAnnotations();
  };

  const handleMouseOver = (lineIdx) => {
    if (displayMode != "annotations" && displayMode != "loss")
      return;

    if (startDragRow !== null) {
      updateSelection(lineIdx);
    }
  };

  const updateSelection = (lineIdx) => {
    const start = Math.min(lineIdx, startDragRow);
    const end = Math.max(lineIdx, startDragRow);
    const draggedIds = new Set(Array.from({ length: end - start + 1 }, (_, i) => i + start));

    let newLines;
    if (isDragSelecting){
      newLines = lines.map((line) => {
        let newLine = { ...line };
        newLine.isSelected = newLine.isSelected || draggedIds.has(newLine.id);
        return newLine;
      });
    } else {
      // drag unselecting
      newLines = lines.map((line) => {
        let newLine = { ...line };
        newLine.isSelected = newLine.isSelected && !draggedIds.has(newLine.id);
        return newLine;
      });
    }
    setLines(newLines);
  };

  const saveAnnotations = () => {
    console.log("Saving annotations...");
    fetch(`http://localhost:8000/example/${id}`, {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        selected_lines: lines.filter((line) => line.isSelected).map((line) => line.id),
        is_ignored: isIgnored
      })
    }).catch(error => {
        alert("Error: " + error);
    });
  };

  const getBgColor = (line) => {
    switch (displayMode) {
      case "annotations":
        return line.isSelected ? "lightgreen" : "#f0f0f0";
      case "predictions":
        return line.predictionScore > threshold ? "lightgreen" : "#f0f0f0";
      case "loss":
        var loss = - (1 * line.isSelected * Math.log(line.predictionScore + 1e-5) + (1 - line.isSelected) * Math.log(1 - line.predictionScore + 1e-5));      
        var interpol = (1 -  Math.max(0, Math.min(2, loss)) / 2); // 1 = max loss intensity, 0 = min loss intensity
        if (line.isSelected) {
          // Loss for selected lines are blue color
          var p1 = [223, 236, 245];
          var p2 = [50, 155, 227];
        } else {
          // Loss for unselected lines are red color 
          var p1 = [247, 227, 225];
          var p2 = [230, 88, 73];
        }
        var redHex = Math.round( p1[0] * interpol + p2[0] * (1-interpol)).toString(16).padStart(2, '0');
        var greenHex = Math.round( p1[1] * interpol + p2[1] * (1-interpol)).toString(16).padStart(2, '0');
        var blueHex = Math.round( p1[2] * interpol + p2[2] * (1-interpol)).toString(16).padStart(2, '0');

        return "#" + redHex + greenHex + blueHex;
    } 
  }

  return (
      <Box>

        {/* Document */}
        <Box width="49%" margin={"10px"} position={"absolute"} top={"200px"}>
          <VStack spacing={0} align="stretch" top={"500px"}>
            {
              (["annotations", "predictions", "loss"].includes(displayMode)) && (
                lines.map((line, index) => (
                  <Box key={index} className="annotation row" 
                      onMouseDown={() => handleMouseDown(line.id)} 
                      onMouseUp={handleMouseUp} 
                      onMouseOver={() => handleMouseOver(line.id)} 
                      bg={getBgColor(line)}
                    >
                    <code>{line.text}</code>
                  </Box>
                ))
              )
            }{
              (displayMode == "output") && (
                lines.filter((line) => line.predictionScore >= threshold).map((line, index) => (
                  <Box key={index} className="row">
                    <code>{line.text}</code>
                  </Box>
                ))
              )
            }
          </VStack>
          <HStack marginTop={10}>
            <Button onClick={() => {navigate(`/${Math.max(0, Number(id)-1)}`); window.scrollTo(0, 0);}}>Previous</Button>
            <Text>{id} / {numExamples}</Text>
            <Button onClick={() => {navigate(`/${Math.min(numExamples, Number(id)+1)}`); window.scrollTo(0, 0);}}>Next</Button>
          </HStack>
        </Box>

        {/* Header */}
        <Box position={"fixed"} background={"white"} width={"100%"} padding={"20px"} top={0}>
          <Heading mb={5}>Text Annotation Tool</Heading>
          <HStack spacing={7}>
            <Button onClick={() => navigate(`/${Math.max(0, Number(id)-1)}`)}>Previous</Button>
            <Text>{id} / {numExamples}</Text>
            <Button onClick={() => navigate(`/${Math.min(numExamples, Number(id)+1)}`)}>Next</Button>
            <Checkbox size='lg' colorScheme='orange' isChecked={isIgnored} onChange={(e) => {setIsIgnored(e.target.checked); console.log(e.target.checked);}}>
              Ignored
            </Checkbox>
            <Select width={200} onChange={(e) => setDisplayMode(e.target.value)}>
              <option value='annotations'>Annotations</option>
              <option value='predictions'>Predictions</option>
              <option value='loss'>Loss</option>
              <option value='output'>Output</option>
            </Select>
            {
              (["predictions", "output"].includes(displayMode) && (
                <Slider onChange={(val) => setThreshold(val)} min={0.0} max={1.0} width="200px" step={0.01}>
                  <SliderMark value={threshold} textAlign='center' bg='blue' color='white' mt='-10' ml='-5' w='12'>
                    {threshold}
                  </SliderMark>
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb /> 
                </Slider>
              ))
            }{
              (displayMode == "loss") && (
                <p><Text>Blue = Annotated positive</Text> <Text>Red = Annotated negative</Text> </p>
              )
            }
          </HStack>
          <Link href={url} target="_blank">{url}</Link>
        </Box>

        <Box  style={{position: "fixed", left: "50%", height: "100%", width: "50%", top: 0}}>
          <iframe src={url} width="100%" height="100%">
          </iframe>
        </Box>
    </Box>
  );
};

export default Index;
