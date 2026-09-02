import SwiftUI

struct RiderSelectionStatus: View {
    let state: RiderSelectionState
    let selectedTimestamp: String?
    let onChange: () -> Void
    let onTryAgain: () -> Void
    let onReselect: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            icon
                .frame(width: 40, height: 40)
                .background(tint.opacity(0.09), in: Circle())

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(titleColor)

                if let subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(Color(.secondaryLabel))
                        .lineLimit(1)
                }
            }

            Spacer(minLength: 4)

            if let actionTitle {
                Button(actionTitle, action: action)
                    .font(.footnote.weight(.medium))
                    .buttonStyle(.plain)
                    .foregroundStyle(Color.slopePrimary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .background(tint.opacity(backgroundOpacity), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(tint.opacity(borderOpacity), lineWidth: 1)
        }
        .animation(.easeInOut(duration: 0.2), value: state)
    }

    @ViewBuilder
    private var icon: some View {
        switch state {
        case .selecting:
            ProgressView()
                .tint(tint)
        case .idle:
            Image(systemName: "person")
                .foregroundStyle(tint)
        case .selected:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(tint)
        case .failed:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(tint)
        case .lost:
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(tint)
        }
    }

    private var title: String {
        switch state {
        case .idle: "No rider selected"
        case .selecting: "Selecting rider..."
        case .selected: "Rider selected"
        case .failed: "Couldn't identify this rider"
        case .lost: "Rider lost"
        }
    }

    private var subtitle: String? {
        switch state {
        case .idle: "Scrub to a clear frame and tap a skier"
        case .selecting: "Analyzing frame"
        case .selected: selectedTimestamp.map { "Selected at \($0)" }
        case .failed: "Tap a clearer part of the skier"
        case .lost: "Scrub to a clear frame and select again"
        }
    }

    private var actionTitle: String? {
        switch state {
        case .selected: "Change"
        case .failed: "Try Again"
        case .lost: "Re-select"
        case .idle, .selecting: nil
        }
    }

    private var action: () -> Void {
        switch state {
        case .selected: onChange
        case .failed: onTryAgain
        case .lost: onReselect
        case .idle, .selecting: {}
        }
    }

    private var tint: Color {
        switch state {
        case .idle: Color(.systemGray3)
        case .selecting, .selected: .slopePrimary
        case .failed: .slopeDanger
        case .lost: .slopeWarning
        }
    }

    private var titleColor: Color {
        state == .idle ? Color(.systemGray3) : Color.primary
    }

    private var backgroundOpacity: Double {
        state == .idle ? 0.08 : 0.05
    }

    private var borderOpacity: Double {
        state == .idle ? 0.15 : 0.14
    }
}

