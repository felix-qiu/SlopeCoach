import XCTest

final class RiderSelectionFlowUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
        app.buttons["primary-button-Start Analysis"].tap()
        XCTAssertTrue(videoPreview.waitForExistence(timeout: 2))
    }

    func testSelectionChangeAndFailureFlow() {
        addScreenshot(named: "01-rider-selection-idle")

        videoPreview
            .coordinate(withNormalizedOffset: CGVector(dx: 0.735, dy: 0.59))
            .tap()

        XCTAssertTrue(app.staticTexts["Rider selected"].waitForExistence(timeout: 2))
        XCTAssertTrue(app.buttons["primary-button-Continue"].isEnabled)
        addScreenshot(named: "02-rider-selection-selected")

        app.buttons["primary-button-Continue"].tap()
        XCTAssertTrue(app.staticTexts["Analyzing..."].waitForExistence(timeout: 2))
        app.buttons["Back"].tap()
        XCTAssertTrue(app.staticTexts["Rider selected"].waitForExistence(timeout: 1))

        app.buttons["Change"].tap()
        XCTAssertTrue(app.staticTexts["No rider selected"].waitForExistence(timeout: 1))
        XCTAssertFalse(app.buttons["primary-button-Continue"].isEnabled)

        videoPreview
            .coordinate(withNormalizedOffset: CGVector(dx: 0.18, dy: 0.2))
            .tap()

        XCTAssertTrue(app.staticTexts["Couldn't identify this rider"].waitForExistence(timeout: 2))
        XCTAssertFalse(app.buttons["primary-button-Continue"].isEnabled)
        addScreenshot(named: "03-rider-selection-failed")

        app.buttons["Try Again"].tap()
        XCTAssertTrue(app.staticTexts["No rider selected"].waitForExistence(timeout: 1))
    }

    private var videoPreview: XCUIElement {
        app.descendants(matching: .any)["Video preview"].firstMatch
    }

    private func addScreenshot(named name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}

